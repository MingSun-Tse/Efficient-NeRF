import os, sys, copy, numpy as np, time, random, json, math
import imageio
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from model.nerf_raybased import NeRF, NeRF_v2, NeRF_v3, NeRF_v4, NeRF_v5
from model.enhance_cnn import EDSR
from run_nerf_raybased_helpers import sample_pdf, ndc_rays, get_rays, get_embedder
from run_nerf_raybased_helpers import parse_expid_iter, to_tensor, to_array, mse2psnr, to8b, img2mse, load_weights_v2, get_selected_coords
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, BlenderDataset, BlenderDataset_v2, get_novel_poses
from ptflops import get_model_complexity_info
from pruner import pruner_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

# set up logging directories -------
from logger import Logger
from utils import Timer, LossLine, PresetLRScheduler, strdict_to_dict, _weights_init_orthogonal, get_n_params_, get_n_flops_
from option import args

logger = Logger(args)
print = logger.log_printer.logprint
accprint = logger.log_printer.accprint
netprint = logger.log_printer.netprint
ExpID = logger.ExpID

# redefine get_rays
from functools import partial
get_rays1 = get_rays
get_rays = partial(get_rays, trans_origin=args.trans_origin, focal_scale=args.focal_scale)

dim_dir = dim_rgb = 3
if args.model_name in ['nerf_v4']:
    dim_dir = dim_rgb = 6
# ---------------------------------

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # @mst: shape: torch.Size([65536, 3]), 65536=1024*64 (n_rays * n_sample_per_ray)
    embedded = embed_fn(inputs_flat) # shape: [n_rays*n_sample_per_ray, 63]

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs) # @mst: train, rays_flat.shape(0) = 1024, chunk = 32768
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) # @mst: 'rays_d' is real-world data, needs normalization.
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float() # @mst: test: [160000, 3], 400*400; train: [1024, 3]
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = int(H / render_factor)
        W = int(W / render_factor)
        focal = focal / render_factor

    # --------------------------------------------------------------------------------------
    # Improve parallism
    # # get all rays
    # rays_o, rays_d = [], []
    # n_pose = len(render_poses)
    # for i, c2w in enumerate(render_poses):
    #     rays_o_, rays_d_ = get_rays(H, W, focal, c2w[:3,:4]) # rays_o shape: # [H, W, 3]
    #     rays_o += rays_o_.view(-1, 3)
    #     rays_d += rays_d_.view(-1, 3)
    # rays_o = torch.cat(rays_o, dim=0).cuda() # [n_pose*H*W, 3]
    # rays_d = torch.cat(rays_d, dim=0).cuda() # [n_pose*H*W, 3]
    
    # model = render_kwargs['network_fn'].cuda()
    # perturb = render_kwargs['perturb']
    # rgb, disp = [], []
    # t0 = time.time()
    # for ix in range(0, rays_o.shape[0], chunk):
    #     with torch.no_grad():
    #         rgb_, disp_, *_ = model(rays_o[ix: ix+chunk], rays_d[ix: ix+chunk], perturb=perturb)
    #         rgb += [rgb_]
    #         disp += [disp_]
    #     print(f'{time.time() - t0}s')
    # rgbs, disps = torch.cat(rgb, dim=0), torch.cat(disp, dim=0)
    # rgbs, disps = rgb.view(n_pose, H, W, 3), disp.view(n_pose, H, W, 3)
    # --------------------------------------------------------------------------------------

    rgbs, disps, errors = [], [], []

    # get all rendering poses
    rays_o_all, rays_d_all = [], []
    for i, c2w in enumerate(render_poses):
        rays_o, rays_d = get_rays1(H, W, focal, c2w[:3, :4]) # rays_o shape: # [H, W, 3]
        rays_o_all.append(rays_o)
        rays_d_all.append(rays_d)

    # render frame by frame
    torch.backends.cudnn.benchmark = True
    for i, c2w in enumerate(render_poses):
        t0 = time.time()
        if args.model_name in ['nerf']:
            rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs) 

        elif args.model_name in ['nerf_v2', 'nerf_v3', 'nerf_v4']:
            # get rays
            model = render_kwargs['network_fn']
            perturb = render_kwargs['perturb']
            randix = np.random.permutation(len(rays_o_all))[0]
            # rays_o, rays_d = get_rays1(H, W, focal, c2w[:3, :4]) # rays_o shape: # [H, W, 3]
            rays_o, rays_d = rays_o_all[randix], rays_d_all[randix]
            rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3) # [H*W, 3]
            print(f'[00] {time.time() - t0:.4f}s')

            # network forward
            if args.model_name in ['nerf_v2', 'nerf_v3']:
                with torch.no_grad():
                    rgb_sum = 0
                    for _ in range(args.render_iters):
                        rgb_sum += model(rays_o, rays_d, perturb=perturb)[0]
                    rgb = rgb_sum / args.render_iters
            elif args.model_name in ['nerf_v4']:
                rays_o = torch.reshape(rays_o, (-1, args.num_shared_pixels * 3)) # TODO-@mst: more advanced reshape
                rays_d = torch.reshape(rays_d, (-1, args.num_shared_pixels * 3))
                with torch.no_grad():
                    rgb = model(rays_o, rays_d, rays_d2=None, scale=args.forward_scale, perturb=perturb, test=True)
                rgb = torch.reshape(rgb, (H, W, 3))
            
            print(f'[01] {time.time() - t0:.4f}s')

            # enhance
            if 'network_enhance' in render_kwargs:
                model_enhance = render_kwargs['network_enhance']
                rgb = model_enhance(rgb, h=H, w=W)

            # reshape to image
            rgb = rgb.view(H, W, 3)
            disp = rgb # placeholder, to maintain compability
                
        rgbs.append(rgb)
        disps.append(disp)
        print(f'[02] {time.time() - t0:.4f}s')

        if gt_imgs is not None:
            errors += [(rgb-gt_imgs[i]).abs()]
        
        print(f'[03] {time.time() - t0:.4f}s')

        if savedir is not None:
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(rgbs[-1]))
            if len(errors):
                imageio.imwrite(filename.replace('.png', '_error.png'), to8b(errors[-1]))
        print(f'[04] {time.time() - t0:.4f}s')

        # torch.cuda.empty_cache()

        print(f'[05] {time.time() - t0:.4f}s')
        if i < 5:
            print(f'[{i}] Rendering one frame, time: {time.time()-t0:.4f}s')
        if i > 3: exit()
        print('')

    rgbs = torch.stack(rgbs, dim=0)
    disps = torch.stack(disps, dim=0)
    
    if gt_imgs is not None:
        test_loss = img2mse(rgbs, gt_imgs)
        test_psnr = mse2psnr(test_loss)
        errors = torch.stack(errors, dim=0)
    else:
        test_loss, test_psnr, errors = None, None, None
    return rgbs, disps, test_loss, test_psnr, errors


def create_nerf(args, near, far):
    """Instantiate NeRF's MLP model.
    """
    # set up model
    model_fine = network_query_fn = None
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    grad_vars = []
    if args.model_name in ['nerf']:
        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]
        model = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model.parameters())

        if args.N_importance > 0:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
            grad_vars += list(model_fine.parameters())

        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    elif args.model_name in ['nerf_v2']:
        model = NeRF_v2(args, near, far, print=netprint).to(device)
        if args.freeze_pretrained:
            assert args.pretrained_ckpt
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            print(f'Freeze model!')
        else:
            grad_vars += list(model.parameters())
        
        if args.enhance_cnn == 'EDSR':
            assert args.select_pixel_mode == 'rand_patch'
            model_enhance = EDSR().to(device)
            grad_vars += list(model_enhance.parameters())
        
        if args.init == 'orth':
            act_func = 'relu' # needs changes if the model does not use ReLU as activation func
            model.apply(lambda m: _weights_init_orthogonal(m, act=act_func))
            print(f'Use orth init. Activation func: {act_func}')

    elif args.model_name in ['nerf_v3']:
        model = NeRF_v3(args, near, far).to(device)
        grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v4']:
        model = NeRF_v4(args, near, far).to(device)
        grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v5']:
        model = NeRF_v5(args, near, far).to(device)
        grad_vars += list(model.parameters())

    # in KD, there is a pretrained teacher
    if args.teacher_ckpt:
        teacher_fn = NeRF(D=8, W=256, input_ch=63, output_ch=4, skips=[4], input_ch_views=27, 
            use_viewdirs=args.use_viewdirs).to(device)
        teacher_fine = NeRF(D=8, W=256, input_ch=63, output_ch=4, skips=[4], input_ch_views=27, 
            use_viewdirs=args.use_viewdirs).to(device) # TODO: not use fixed arguments
        
        # set to eval
        teacher_fn.eval()
        teacher_fine.eval()
        for param in teacher_fn.parameters():
            param.requires_grad = False
        for param in teacher_fine.parameters():
            param.requires_grad = False
        
        # load weights
        ckpt = torch.load(args.teacher_ckpt)
        load_weights_v2(teacher_fn, ckpt, 'network_fn_state_dict')
        load_weights_v2(teacher_fine, ckpt, 'network_fine_state_dict')
        print(f'Load teacher ckpt successfully: "{args.teacher_ckpt}"')

        # get network_query_fn
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=embed_fn,
                                                                    embeddirs_fn=embeddirs_fn,
                                                                    netchunk=args.netchunk)
    # set up optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # start iteration
    start = 0

    # load pretrained checkpoint
    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt)
        if 'network_fn' in ckpt:
            model = ckpt['network_fn']
            if model_fine is not None:
                assert 'network_fine' in ckpt
                model_fine = ckpt['network_fine']
            print(f'Use model arch saved in checkpoint: "{args.pretrained_ckpt}"')
        
        # load state_dict
        load_weights_v2(model, ckpt, 'network_fn_state_dict')
        if model_fine is not None:
            load_weights_v2(model_fine, ckpt, 'network_fine_state_dict')
        print(f'Load pretrained ckpt successfully: "{args.pretrained_ckpt}"')
        
        if args.resume:
            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('Resume optimizer successfully')

    # use DataParallel
    model = torch.nn.DataParallel(model)
    if model_fine is not None:
        model_fine = torch.nn.DataParallel(model_fine)

    # pruning, before 'render_kwargs_train'
    if args.model_name in ['nerf_v2', 'nerf_v3'] and args.pruner:
        class passer: pass
        pruner = pruner_dict[args.pruner].Pruner(model, args, logger, passer)
        model = pruner.prune()
        print('Got just pruned model.')

    # set up training args
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # set up testing args
    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = args.perturb_test
    render_kwargs_test['raw_noise_std'] = 0.

    if args.teacher_ckpt:
        render_kwargs_train['teacher_fn'] = teacher_fn
        render_kwargs_train['teacher_fine'] = teacher_fine
    
    if args.enhance_cnn:
        render_kwargs_train['network_enhance'] = model_enhance
        render_kwargs_test['network_enhance'] = model_enhance

    # get FLOPs and params
    netprint(model)
    n_params = get_n_params_(model)
    if args.model_name == 'nerf':
        dummy_input = torch.randn(1, input_ch + input_ch_views).to(device)
        n_flops = get_n_flops_(model, input=dummy_input, count_adds=False) * (args.N_samples + args.N_samples +  args.N_importance)

        # macs, params = get_model_complexity_info(model, dummy_input.shape, as_strings=True,
        #                                         print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
    elif args.model_name in ['nerf_v2', 'nerf_v3']:
        dummy_rays_o = torch.randn(1, 3).to(device)
        dummy_rays_d = torch.randn(1, 3).to(device)
        n_flops = get_n_flops_(model, input=dummy_rays_o, count_adds=False, rays_d=dummy_rays_d)
    
    elif args.model_name in ['nerf_v4']: # TODO-@smt: this is not correct
        dummy_rays_o = torch.randn(1, 3).to(device)
        dummy_rays_d = torch.randn(1, 3).to(device)
        n_flops = get_n_flops_(model, input=dummy_rays_o, count_adds=False, rays_d=dummy_rays_d, rays_d2=dummy_rays_d)
    
    elif args.model_name in ['nerf_v5']:
        input_dim = input_ch + input_ch_views if args.use_viewdirs else input_ch
        n_img, H, W = 100, 400, 400
        dummy_rays_o = torch.ones(n_img, input_dim * args.n_sample_per_ray, H, W)
        t0 = time.time()
        for ix in range(n_img):
            t1 = time.time()
            with torch.no_grad():
                model(dummy_rays_o[ix].unsqueeze(dim=0).cuda())
            t2 = time.time()
            print(f'{ix:2d} Forward time per image: {t2 - t1:.4f}s')
        print(f'Average: {(t2-t0)/n_img:.4f}s')
        exit()
        n_flops = get_n_flops_(model, input=dummy_rays_o, count_adds=False) / (n_img * H * W)

    print(f'Model complexity per pixel: FLOPs {n_flops/1e6:.4f}M, Params {n_params/1e6:.4f}M')
    
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, verbose=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists) # @mst: opacity

    dists = z_vals[...,1:] - z_vals[...,:-1] # dists for 'distances'
    dists = torch.cat([dists, to_tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    # @mst: 1e10 for infinite distance

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1) # @mst: direction vector needs normalization. why this * ?

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3], RGB for each sampled point 
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape).to(device) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = to_tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]

    # print to check alpha
    if verbose and global_step % args.i_print == 0:
        for i_ray in range(0, alpha.shape[0], 100):
            logtmp = ['%.4f' % x  for x in alpha[i_ray]]
            netprint('%4d: ' % i_ray + ' '.join(logtmp))

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1] # @mst: [N_rays, N_samples]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map).to(device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0] # N_rays = 32768 (1024*32) for test, 1024 for train
    # @mst: ray_batch.shape, train: [1024, 11]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each, o for 'origin', d for 'direction'
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # @mst: near=2, far=6, in batch

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples]) # @mst: shape: torch.Size([1024, 64]) for train, torch.Size([32768, 64]) for test

    # @mst: perturbation of depth z, with each depth value at the middle point
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device) # uniform dist [0, 1)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = to_tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand
    
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    # when training: [1024, 1, 3] + [1024, 1, 3] * [1024, 64, 1]
    # rays_d range: [-1, 1]

#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, verbose=verbose)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid.cpu(), weights[...,1:-1].cpu(), N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach().to(device)

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1) # @mst: sort to merge the fine samples with the coarse samples
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

# Ray helpers
def get_rays(H, W, focal, c2w, trans_origin='', focal_scale=1):
    focal *= focal_scale
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)) # pytorch's meshgrid has indexing='ij'
    i, j = i.t().to(device), j.t().to(device)
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1) # TODO-@mst: check if this H/W or W/H is in the right order
    # Rotate ray directions from camera frame to the world frame
    # rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(dim=-2) * c2w[:3,:3], -1)  # shape: [H, W, 3]
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_pseudo_ratio(schedule, current_step):
    '''example of schedule: 1:0.2,500000:0.9'''
    steps, prs = [], []
    for item in schedule.split(','):
        step, pr = item.split(':')
        step, pr = int(step), float(pr)
        steps += [step]
        prs += [pr]
    
    # linear scheduling
    if current_step < steps[0]:
        pr = prs[0]
    elif current_step > steps[1]:
        pr = prs[1]
    else:
        pr = (prs[1] - prs[0]) / (steps[1] - steps[0]) * (current_step - steps[0]) + prs[0]
    return pr

def train():
    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, poses.shape, render_poses.shape, hwf, args.datadir)
        # Loaded blender (138, 400, 400, 4) (138, 4, 4) torch.Size([40, 4, 4]) [400, 400, 555.5555155968841] ./data/nerf_synthetic/lego
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    
    # Create log dir and copy the config file
    f = f'{logger.log_path}/args.txt'
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = f'{logger.log_path}/config.txt'
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, near, far)
    print(f'Created model {args.model_name}')

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    k = math.sqrt(float(args.N_rand) / H / W)
    patch_h, patch_w = int(H * k), int(W * k)

    # get train, test, video poses and images
    train_images, train_poses = images[i_train], poses[i_train]
    test_poses, test_images = poses[i_test], images[i_test]
    n_original_img = len(train_images)
    if args.dataset_type == 'blender':
        video_poses = get_novel_poses(args, n_pose=args.n_pose_video)
    else:
        video_poses = render_poses
    
    video_poses = get_novel_poses(args, n_pose=args.n_pose_video)

    
    if args.model_name in ['nerf']:
        # set up input
        t0 = time.time()

        t1 = time.time()
        if args.

    
    if args.model_name in ['nerf_v2', 'nerf_v3']:


    
    
    if args.model_name in ['nerf_v4']:
        pass
    
    
    
    
    
    if args.model_name in ['nerf_v5']:



        

if __name__=='__main__':
    train()
