import os, sys, copy, numpy as np, time, random, json, math
import imageio
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
# from torch.utils.tensorboard import SummaryWriter
from model.nerf_raybased import NeRF, NeRF_v2, NeRF_v3, NeRF_v3_2, NeRF_v3_3, NeRF_v3_4, NeRF_v3_5, NeRF_v3_6, NeRF_v3_7, NeRF_v4, NeRF_v6
from model.nerf_raybased import NeRF_v3_8
from model.nerf_raybased import PositionalEmbedder, PointSampler
from model.enhance_cnn import EDSR
from run_nerf_raybased_helpers import sample_pdf, ndc_rays, get_rays, get_embedder
from run_nerf_raybased_helpers import parse_expid_iter, to_tensor, to_array, mse2psnr, to8b, img2mse, load_weights_v2, get_selected_coords
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, BlenderDataset, BlenderDataset_v2, BlenderDataset_v3, BlenderDataset_v4, get_novel_poses
from ptflops import get_model_complexity_info
from pruner import pruner_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

# set up logging directories -------
from logger import Logger
from utils import Timer, LossLine, PresetLRScheduler, strdict_to_dict, _weights_init_orthogonal, get_n_params_, get_n_flops_
from utils import AverageMeter, ProgressMeter
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

DIM_DIR = args.dim_dir
DIM_RGB = args.dim_rgb
if args.model_name in ['nerf_v4']:
    DIM_DIR = DIM_RGB = 6

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
# ---------------------------------
def square_rand_bbox(img_h, img_w, rand_crop_size):
    bbx1 = np.random.randint(0, img_w - rand_crop_size + 1)
    bby1 = np.random.randint(0, img_h - rand_crop_size + 1)
    return bbx1, bby1, bbx1 + rand_crop_size, bby1 + rand_crop_size

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
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    render_kwargs['network_fn'].eval()
    rgbs, disps, errors, model_inputs = [], [], [], []
    raybased_nerf = ['nerf_v2', 'nerf_v3', 'nerf_v3.2', 'nerf_v3.3', 'nerf_v3.4', 'nerf_v3.4.2', 
            'nerf_v3.6', 'nerf_v3.7', 'nerf_v3.5', 'nerf_v3.8', 'nerf_v4', 'nerf_v6']
            
    for i, c2w in enumerate(render_poses):
        torch.cuda.synchronize()
        t0 = time.time()
        print(f'[#{i}] frame, rendering begins')
        if args.model_name in ['nerf']:
            rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs) 

        elif args.model_name in raybased_nerf:
            model = render_kwargs['network_fn']
            perturb = render_kwargs['perturb']

            # get rays_o and rays_d
            if args.model_name in ['nerf_v2', 'nerf_v3', 'nerf_v4', 'nerf_v3.4.2', 'nerf_v3.6', 'nerf_v3.7', 'nerf_v3.8']:
                rays_o, rays_d = get_rays1(H, W, focal, c2w[:3, :4]) # [H, W, 3]
                rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3) # [H*W, 3]

            # network forward
            if args.model_name in ['nerf_v2', 'nerf_v3']:
                with torch.no_grad():
                    rgb = 0
                    for _ in range(args.render_iters):
                        rgb += model(rays_o, rays_d, perturb=perturb)[0]
                    rgb /= args.render_iters
            
            elif args.model_name in ['nerf_v3.2']:
                with torch.no_grad():
                    model_input = positional_embedder(point_sampler.sample_test(c2w))
                    rgb = model(model_input)
                    # model_inputs += [model_input]
            
            elif args.model_name in ['nerf_v3.3', 'nerf_v3.5']:
                with torch.no_grad():
                    model_input = positional_embedder(point_sampler.sample_test(c2w))
                    rgb = model.forward_mlp2(model_input, img_h=H, img_w=W)

            elif args.model_name in ['nerf_v3.4']:
                with torch.no_grad():
                    pts = point_sampler.sample_test2(c2w) # [H*W, n_sample, 3]
                    pts = pts.view(H, W, args.n_sample_per_ray, 3) # [H, W, n_sample, 3]
                    pts = positional_embedder.embed(pts) # [H, W, n_sample, 3, 2L+1]
                    pts = pts.view(*pts.shape[:2], -1) # [H, W, -1]
                    patch_size = args.scale
                    num_h, num_w = H // patch_size, W // patch_size
                    model_input = []
                    torch.cuda.synchronize()
                    t1 = time.time()
                    for h_ix in range(num_h):
                        for w_ix in range(num_w):
                             patch = pts[h_ix*patch_size: (h_ix+1)*patch_size, w_ix*patch_size: (w_ix+1)*patch_size, ...] # [3, 3, -1]
                             model_input += [patch]
                    model_input = torch.stack(model_input, dim=0) # [n_patch, 3, 3 , -1]
                    model_input = model_input.view(model_input.shape[0], -1)
                    torch.cuda.synchronize()
                    print(f'data prepare {time.time()-t1:.4f}s')
                    # model_input = torch.ones((H//args.scale)*(W//args.scale), args.scale*args.scale*args.n_sample_per_ray*3*21).to(device)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    rgb = model(model_input) # [n_patch, 27]
                    torch.cuda.synchronize()
                    print(f'model forward {time.time() - t1:.4f}s')

            elif args.model_name in ['nerf_v3.4.2', 'nerf_v3.6', 'nerf_v3.7']:
                with torch.no_grad():
                    patch_size = args.scale
                    rays_o, rays_d = rays_o.view(H, W, 3), rays_d.view(H, W, 3)
                    num_h, num_w = H // patch_size, W // patch_size
                    
                    torch.cuda.synchronize()
                    t11 = time.time()
                    model_input = []
                    for h_ix in range(num_h):
                        for w_ix in range(num_w):
                            rays_o_patch = rays_o[h_ix*patch_size: (h_ix+1)*patch_size, w_ix*patch_size: (w_ix+1)*patch_size] # [3, 3, 3]
                            rays_d_patch = rays_d[h_ix*patch_size: (h_ix+1)*patch_size, w_ix*patch_size: (w_ix+1)*patch_size] # [3, 3, 3]
                            rays_o_patch = rays_o_patch.reshape([1, -1]) # [1, 27]
                            rays_d_patch = rays_d_patch.reshape([1, -1]) # [1, 27]
                            pts = point_sampler.sample_train(rays_o_patch, rays_d_patch, perturb=0.) # [1, 9*n_sample*3]
                            pts = positional_embedder(pts)[0] # [1, 9*n_sample*3*embed_dim]
                            model_input += [pts]
                    torch.cuda.synchronize()
                    print(f'prepare input: {time.time() - t11:.4f}s')
                    
                    # forward in batch    
                    model_input = torch.stack(model_input, dim=0) # [n_ray, 9*n_sample*3*embed_dim]
                    torch.cuda.synchronize()
                    t11 = time.time()
                    rgb_patches = model(model_input) # [n_ray, 27]

                    torch.cuda.synchronize()
                    print(f'model forward: {time.time() - t11:.4f}s')

                    torch.cuda.synchronize()
                    t11 = time.time()
                    rgb, cnt = torch.zeros(num_h * patch_size, num_w * patch_size, 3).cuda(), -1
                    for h_ix in range(num_h):
                        for w_ix in range(num_w):
                            cnt += 1
                            rgb_ = rgb_patches[cnt].view(patch_size, patch_size, 3) # [3, 3, 3]
                            rgb[h_ix*patch_size: (h_ix+1)*patch_size, w_ix*patch_size: (w_ix+1)*patch_size, :] = rgb_

                    torch.cuda.synchronize()
                    print(f'prepare output: {time.time() - t11:.4f}s')

            elif args.model_name in ['nerf_v3.8']:
                with torch.no_grad():
                    patch_size = 16
                    rays_o, rays_d = rays_o.view(H, W, 3), rays_d.view(H, W, 3)
                    num_h, num_w = H // patch_size, W // patch_size
                    
                    torch.cuda.synchronize()
                    t11 = time.time()
                    rays_o_patch, rays_d_patch = [], []
                    for h_ix in range(num_h):
                        for w_ix in range(num_w):
                            rays_o_patch += [rays_o[h_ix*patch_size: (h_ix+1)*patch_size, w_ix*patch_size: (w_ix+1)*patch_size]] # [patch_size, patch_size, 3]
                            rays_d_patch += [rays_d[h_ix*patch_size: (h_ix+1)*patch_size, w_ix*patch_size: (w_ix+1)*patch_size]] # [patch_size, patch_size, 3]
                    rays_o_patch = torch.stack(rays_o_patch, dim=0) # [n_patch, patch_size, patch_size, 3]
                    rays_d_patch = torch.stack(rays_d_patch, dim=0) # [n_patch, patch_size, patch_size, 3]
                    pts = point_sampler.sample_train_cnnstyle(rays_o_patch, rays_d_patch, perturb=0.) # [n_patch, patch_size, patch_size, 16, 3]
                    pts = positional_embedder.embed_cnnstyle(pts) # [n_patch, patch_size, patch_size, 16, 3, 2L+1]
                    pts = pts.view(*pts.shape[:3], -1)
                    model_input = pts.permute(0, 3, 1, 2)
                    torch.cuda.synchronize()
                    print(f'prepare input: {time.time() - t11:.4f}s')
                    
                    # forward in batch    
                    torch.cuda.synchronize()
                    t11 = time.time()
                    rgb_patches = model(model_input) # [n_patch, 3, patch_size, patch_size]
                    torch.cuda.synchronize()
                    print(f'model forward: {time.time() - t11:.4f}s')

                    torch.cuda.synchronize()
                    t11 = time.time()
                    rgb, cnt = torch.zeros(num_h * patch_size, num_w * patch_size, 3).cuda(), -1
                    for h_ix in range(num_h):
                        for w_ix in range(num_w):
                            cnt += 1
                            rgb_ = rgb_patches[cnt].permute(1, 2, 0) # [patch_size, patch_size, 3]
                            rgb[h_ix*patch_size: (h_ix+1)*patch_size, w_ix*patch_size: (w_ix+1)*patch_size, :] = rgb_
                    torch.cuda.synchronize()
                    print(f'prepare output: {time.time() - t11:.4f}s')

            elif args.model_name in ['nerf_v4']:
                rays_o = torch.reshape(rays_o, (-1, args.num_shared_pixels * 3))
                rays_d = torch.reshape(rays_d, (-1, args.num_shared_pixels * 3))
                with torch.no_grad():
                    rgb = model(rays_o, rays_d, rays_d2=None, scale=args.forward_scale, perturb=perturb, test=True)
                rgb = torch.reshape(rgb, (H, W, 3))
            
            elif args.model_name in ['nerf_v6']:
                model_input = positional_embedder(point_sampler.sample_test(c2w)) # [n_ray, embed_dim]
                model_input = model_input.view(1, H, W, model_input.shape[-1])
                model_input = model_input.permute(0, 3, 1, 2)
                # model_input = model_input.view(1, model_input.shape[-1], H, W) # Note! mind the meaning of each axis
                with torch.no_grad():
                    rgb = model(model_input) # [1, 3, H, W]
                    rgb = rgb.permute(0, 2, 3, 1) # [1, H, W, 3]

            # enhance
            if 'network_enhance' in render_kwargs:
                model_enhance = render_kwargs['network_enhance']
                if len(rgb.shape) == 3:
                    rgb = rgb[None, ...] # [1, H, W, 3]
                rgb = rgb.permute(0, 3, 1, 2)
                rgb = model_enhance(rgb) # [1, 3, H, W]
                rgb = rgb.permute(0, 2, 3, 1) # [1, H, W, 3]

            # reshape to image
            H_ = W_ = int(math.sqrt(rgb.numel() / 3)) # TODO-@mst: may not be square
            rgb = rgb.view(H_, W_, 3)
            disp = rgb # placeholder, to maintain compability
  
        rgbs.append(rgb)
        disps.append(disp)

        if gt_imgs is not None:
            errors += [(rgb - gt_imgs[i][:H_, :W_, :]).abs()]
        
        if savedir is not None:
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(rgbs[-1]))
            if len(errors):
                imageio.imwrite(filename.replace('.png', '_error.png'), to8b(errors[-1]))
        
        torch.cuda.synchronize()
        print(f'[#{i}] frame, rendering done, time for this frame: {time.time()-t0:.4f}s')
        print('')
    
    rgbs = torch.stack(rgbs, dim=0)
    disps = torch.stack(disps, dim=0)
    ## save for nerf_v3.2, will be removed
    # np.save('model_input.npy', torch.stack(model_inputs, dim=0).data.cpu().numpy()[5])
    # np.save('model_output.npy', rgbs.data.cpu().numpy()[5])
    # filename = 'model_output.png'
    # imageio.imwrite(filename, to8b(rgbs[5]))
    
    if gt_imgs is not None:
        test_loss = img2mse(rgbs, gt_imgs[:, :H_, :W_])
        test_psnr = mse2psnr(test_loss)
        errors = torch.stack(errors, dim=0)
    else:
        test_loss, test_psnr, errors = None, None, None
    
    render_kwargs['network_fn'].train()
    return rgbs, disps, test_loss, test_psnr, errors


def render_func(model, pose):
    with torch.no_grad():
        rgb = model(positional_embedder(point_sampler.sample_test(pose)))
    return rgb

def create_nerf(args, near, far):
    """Instantiate NeRF's MLP model.
    """
    # set up model
    model_fine = network_query_fn = None
    global embed_fn
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    # @mst: use external positional embedding for our raybased nerf
    global positional_embedder; positional_embedder = PositionalEmbedder(L=args.multires)

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
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())
        
        if args.init == 'orth':
            act_func = 'relu' # needs changes if the model does not use ReLU as activation func
            model.apply(lambda m: _weights_init_orthogonal(m, act=act_func))
            print(f'Use orth init. Activation func: {act_func}')

    elif args.model_name in ['nerf_v3']:
        model = NeRF_v3(args, near, far).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())
    
    elif args.model_name in ['nerf_v3.2']:
        input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v3_2(args, input_dim).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v3.3']:
        input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v3_3(args, input_dim).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v3.4', 'nerf_v3.4.2']:
        input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v3_4(args, input_dim, scale=args.scale).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v3.6']:
        input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v3_6(args, input_dim, scale=args.scale).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v3.7']:
        input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v3_7(args, input_dim, scale=args.scale).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v3.8']:
        input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v3_8(args, input_dim).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v3.5']:
        input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v3_5(args, input_dim, scale=args.scale).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v4']:
        model = NeRF_v4(args, near, far).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())

    elif args.model_name in ['nerf_v6']:
        input_dim = args.n_sample_per_ray * 3 * positional_embedder.embed_dim
        model = NeRF_v6(args, input_dim).to(device)
        if not args.freeze_pretrained:
            grad_vars += list(model.parameters())
    
    # freeze pretrained model
    if args.freeze_pretrained:
        assert args.pretrained_ckpt
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f'Freeze model!')
    
    # enhance cnn
    if args.enhance_cnn == 'EDSR':
        model_enhance = EDSR().to(device)
        grad_vars += list(model_enhance.parameters())

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
    if not args.render_only: # when rendering, use just one GPU
        model = MyDataParallel(model)
        if model_fine is not None:
            model_fine = MyDataParallel(model_fine)
        if hasattr(model.module, 'input_dim'): model.input_dim = model.module.input_dim

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

    elif args.model_name in ['nerf_v3.2']:
        dummy_input = torch.randn(1, model.input_dim).to(device)
        n_flops = get_n_flops_(model, input=dummy_input, count_adds=False)

    elif args.model_name in ['nerf_v3.4', 'nerf_v3.4.2', 'nerf_v3.6', 'nerf_v3.7']:
        dummy_input = torch.randn(1, model.input_dim * (args.scale ** 2)).to(device)
        n_flops = get_n_flops_(model, input=dummy_input, count_adds=False) / (args.scale ** 2)

    # elif args.model_name in ['nerf_v3.5']:
    #     n_img, H, W = 1, 400, 400
    #     dummy_input = torch.randn(n_img, model.input_dim, H, W).to(device) # CNN-style input
    #     n_flops = get_n_flops_(model, input=dummy_input, count_adds=False) / (n_img * H * W)

    elif args.model_name in ['nerf_v3.3', 'nerf_v6']:
        n_img, H, W = 1, 400, 400
        dummy_input = torch.randn(n_img, model.input_dim, H, W).to(device) # CNN-style input
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model(dummy_input)
        torch.cuda.synchronize()
        print(f'Forward time per image: {time.time() - t0:.4f}s')
        n_flops = get_n_flops_(model, input=dummy_input, count_adds=False) / (n_img * H * W)

    elif args.model_name in ['nerf_v3.8']:
        n_img, H, W = 1, 16, 16
        dummy_input = torch.randn(n_img, model.input_dim, H, W).to(device) # CNN-style input
        n_flops = get_n_flops_(model, input=dummy_input, count_adds=False) / (n_img * H * W)

    elif args.model_name in ['nerf_v4']: # TODO-@smt: this is not correct
        dummy_rays_o = torch.randn(1, 3).to(device)
        dummy_rays_d = torch.randn(1, 3).to(device)
        n_flops = get_n_flops_(model, input=dummy_rays_o, count_adds=False, rays_d=dummy_rays_d, rays_d2=dummy_rays_d)
    
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

def get_teacher_targets(poses, H, W, focal, render_kwargs_train, args):
    render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
    render_kwargs_['network_fn'] = render_kwargs_train['teacher_fn'] # temporarily change the network_fn
    render_kwargs_['network_fine'] = render_kwargs_train['teacher_fine'] # temporarily change the network_fine
    render_kwargs_.pop('teacher_fn')
    render_kwargs_.pop('teacher_fine')
    teacher_target = []
    t_ = time.time()
    for ix, pose in enumerate(poses):
        print(f'[{ix}/{len(poses)}] Using teacher to render more images... elapsed time: {(time.time() - t_):.2f}s')
        rays_o, rays_d = get_rays(H, W, focal, pose)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        rgb, *_ = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                        verbose=False, retraw=False,
                                        **render_kwargs_)
        teacher_target.append(rgb)
        # # check pseudo images
        # filename = f'kd_fern_{ix}.png'
        # imageio.imwrite(filename, to8b(rgb.data.cpu().numpy()))
    print(f'Teacher rendering done ({len(poses)} views). Time: {(time.time() - t_):.2f}s')
    return teacher_target

def get_teacher_targets_v2(poses, H, W, focal, render_kwargs_train, args, pose_tag, check_img=False):
    '''Save teacher target as .npy
    '''
    teacher_expid, teacher_iter = parse_expid_iter(args.teacher_ckpt)
    target_path = f'data/pose_targets_{teacher_expid}_iter{teacher_iter}_pose{pose_tag}.npy'
    if os.path.exists(target_path):
        rgbs = np.load(target_path)
        rgbs = to_tensor(rgbs)
        print(f'Loaded saved teacher targets: "{target_path}"')
        return rgbs
    else:
        if args.debug:
            factor = args.render_factor if args.render_factor != 0 else 1
            return torch.randn([poses.shape[0], int(H/factor), int(W/factor), 3]).to(device)
        hwf = [H, W, focal]
        render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
        render_kwargs_['network_fn'] = render_kwargs_train['teacher_fn'] # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train['teacher_fine'] # temporarily change the network_fine
        render_kwargs_.pop('teacher_fn')
        render_kwargs_.pop('teacher_fine')
        rgbs, *_ = render_path(poses, hwf, args.chunk, render_kwargs_, render_factor=args.render_factor)
        np.save(target_path, to_array(rgbs))

        # check teacher images
        if check_img:
            savedir = target_path.replace('.npy', '')
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            for ix, rgb in enumerate(rgbs):
                imageio.imwrite(f'{savedir}/{ix}.png', to8b(rgb))
        return rgbs

def InfiniteSampler(n):
    order = np.random.permutation(n)
    i = 0
    while True:
        yield order[i]
        i += 1
        if i == n:
            order = np.random.permutation(n)
            i = 0

from torch.utils import data
class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples
    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))
    def __len__(self):
        return 2 ** 31

def get_dataloader(dataset_type, datadir, pseudo_ratio=0.5):
    if dataset_type == 'blender':
        if args.data_mode in ['images']:
            trainset = BlenderDataset(datadir, pseudo_ratio)
            trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                    batch_size=1,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    sampler=InfiniteSamplerWrapper(len(trainset))
            )
        elif args.data_mode in ['rays', '16x16patches_v3']:
            trainset = BlenderDataset_v2(datadir, dim_dir=DIM_DIR, dim_rgb=DIM_RGB)
            trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                    batch_size=args.N_rand,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    sampler=InfiniteSamplerWrapper(len(trainset))
            )
        elif args.data_mode in ['images_new']:
            trainset = BlenderDataset_v2(datadir, rand_crop_size=args.rand_crop_size, img_H=IMG_H, img_W=IMG_W)
            trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                    batch_size=args.N_rand,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    sampler=InfiniteSamplerWrapper(len(trainset))
            )
        elif args.data_mode in ['16x16patches']:
            trainset = BlenderDataset_v3(datadir)
            trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                    batch_size=args.N_rand,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    sampler=InfiniteSamplerWrapper(len(trainset))
            )
        elif args.data_mode in ['16x16patches_v2']:
            trainset = BlenderDataset_v4(datadir)
            trainloader = torch.utils.data.DataLoader(dataset=trainset, 
                    batch_size=args.N_rand,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    sampler=InfiniteSamplerWrapper(len(trainset))
            )
    return iter(trainloader), len(trainset)

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

def save_onnx(model, onnx_path):
    model = copy.deepcopy(model)
    mobile_H = mobile_W = 256
    if hasattr(model, 'module'): model = model.module
    if args.model_name in ['nerf_v3.2']:
        dummy_input = torch.randn(mobile_H * mobile_W, model.input_dim).to(device)
    elif args.model_name in ['nerf_v3.3', 'nerf_v6']:
        dummy_input = torch.randn(1, model.input_dim, mobile_H, mobile_W).to(device)
    else:
        raise NotImplementedError
    torch.onnx.export(model.cpu(),
                    dummy_input.cpu(),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    })
    del model

#TODO-@mst: move these utility functions to a better place
def check_onnx(model, onnx_path):
    ''' refer to https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html '''
    import onnx, onnxruntime
    model = copy.deepcopy(model)
    if hasattr(model, 'module'): model = model.module
    model = model.cpu()
    # get torch output as ground truth
    batch_size = 64
    x = torch.randn(batch_size, model.input_dim, requires_grad=True)
    torch_out = model(x)
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

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
    global IMG_H; global IMG_W
    IMG_H, IMG_W = H, W

    # set up sampler
    global point_sampler; point_sampler = PointSampler(H, W, focal, args.n_sample_per_ray, near, far)

    # get train, test, video poses and images
    train_images, train_poses = images[i_train], poses[i_train]
    test_poses, test_images = poses[i_test], images[i_test]
    n_original_img = len(train_images)
    if args.dataset_type == 'blender':
        video_poses = get_novel_poses(args, n_pose=args.n_pose_video)
    else:
        video_poses = render_poses
    # # ------------- check poses
    # netprint('=====> Test poses:')
    # for p in test_poses:
    #     netprint(p.data.cpu().numpy())
    #     netprint(f'radius: {(p[:3, -1]).norm().item()}')
    #     diff = torch.zeros((360, 90))
    #     for i in range(360):
    #         for j in range(90):
    #             diff[i, j] = (pose_spherical(i-180, -j, 4.0311289) - p).norm()
    #     netprint(f'its angle: {(diff.argmin()/90-180).item(), (-diff.argmin()%90).item()}, diff: {diff.min().item()}')
    
    # netprint('\n=====> Video poses:')
    # video_poses = get_novel_poses(args, n_pose=args.n_pose_video)
    # for p in video_poses[:20]:
    #     netprint(p.data.cpu().numpy())
    #     netprint(f'radius: {(p[:3, -1]).norm().item()}')
    # exit()
    # # -------------

    # data sketch
    print(f'{len(i_train)} original train views are [{" ".join([str(x) for x in i_train])}]')
    print(f'{len(i_test)} test views are [{" ".join([str(x) for x in i_test])}]')
    print(f'{len(i_val)} val views are [{" ".join([str(x) for x in i_val])}]')
    print(f'train_images shape {train_images.shape} train_poses shape {train_images.shape}')

    if args.test_teacher:
        assert args.teacher_ckpt
        print('Testing teacher...')
        render_kwargs_ = {x: v for x, v in render_kwargs_test.items()}
        render_kwargs_['network_fn'] = render_kwargs_train['teacher_fn'] # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train['teacher_fine'] # temporarily change the network_fine
        with torch.no_grad():
            *_, test_loss, test_psnr, _ = render_path(test_poses, hwf, 4096, render_kwargs_, gt_imgs=test_images, render_factor=args.render_factor)
        print(f'Teacher test: Loss {test_loss.item():.4f} PSNR {test_psnr.item():.4f}')

    if args.test_pretrained:
        print('Testing pretrained...')
        with torch.no_grad():
            *_, test_loss, test_psnr,_  = render_path(test_poses, hwf, 4096, render_kwargs_test, gt_imgs=test_images, render_factor=args.render_factor)
        print(f'Pretrained test: Loss {test_loss.item():.4f} PSNR {test_psnr.item():.4f}')

    # @mst: use dataloader for training
    kd_poses = None
    if args.datadir_kd:
        if args.dataset_type == 'blender':
            pr = get_pseudo_ratio(args.pseudo_ratio_schedule, current_step=start+1)
            trainloader, n_total_img = get_dataloader(args.dataset_type, args.datadir_kd.split(':')[1], pseudo_ratio=pr)
        else: # LLFF dataset
            kd_poses = copy.deepcopy(render_poses)
            print(f'Using teacher to render {len(kd_poses)} images for KD...')
            kd_targets = get_teacher_targets_v2(kd_poses, H, W, focal, render_kwargs_train, args, pose_tag=args.n_pose_kd)
            n_total_img = len(kd_poses) + len(train_images)
            pr = len(kd_poses) / n_total_img
        print(f'Loaded data. Now total #train images: {n_total_img} Pseudo_ratio: {pr:.4f} ')

    # get video_targets
    video_targets = None
    if args.teacher_ckpt:
        t_ = time.time()
        print('Get video targets...')
        if kd_poses is not None and (video_poses - kd_poses).abs().sum() == 0:
            video_targets = kd_targets
        else:
            video_targets = get_teacher_targets_v2(video_poses, H, W, focal, render_kwargs_train, args, pose_tag=args.n_pose_video)
        print(f'Get video targets done (time: {time.time() - t_:.2f}s)')

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        expid, iter_ = parse_expid_iter(args.pretrained_ckpt)
        with torch.no_grad():
            t_ = time.time()
            if args.render_test:
                print('Rendering test images...')
                rgbs, *_, test_loss, test_psnr, errors = render_path(test_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=test_images, savedir=logger.gen_img_path, render_factor=args.render_factor)
                print(f'[TEST] Loss {test_loss.item():.4f} PSNR {test_psnr.item():.4f}')
            else:
                if args.dataset_type == 'blender':
                    video_poses = get_novel_poses(args, n_pose=args.n_pose_video)
                else:
                    video_poses = render_poses
                print(f'Rendering video... (n_pose: {len(video_poses)})')
                rgbs, *_, errors = render_path(video_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=video_targets, render_factor=args.render_factor)
            t = time.time() - t_
        video_path = f'{logger.gen_img_path}/video_{expid}_iter{iter_}_{args.video_tag}.mp4'
        imageio.mimwrite(video_path, to8b(rgbs), fps=30, quality=8)
        if errors is not None:
            imageio.mimwrite(video_path.replace('.mp4', '_error.mp4'), to8b(errors), fps=30, quality=8)
        print(f'Save video: "{video_path} (time: {t:.2f}s)')
        exit(0)
    
    if args.convert_to_onnx:
        onnx_path = args.pretrained_ckpt.replace('.tar', '.onnx')
        save_onnx(render_kwargs_test['network_fn'], onnx_path)
        check_onnx(render_kwargs_test['network_fn'], onnx_path)
        print(f'Convert to onnx done. Saved at "{onnx_path}"') 
        exit(0)

    if args.benchmark:
        x = video_poses[0]
        timer = benchmark.Timer(
            stmt='render_func(model, pose)',
            setup='from __main__ import render_func',
            globals={'model': render_kwargs_test['network_fn'], 'pose': x})
        print(timer.timeit(100))
        exit(0)
    
    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    # @mst: use our own lr scheduler
    if args.lr:
        lr_scheduler = PresetLRScheduler(strdict_to_dict(args.lr, ttype=float))

    if args.hard_ratio:
        hard_rays = to_tensor([])

    # training
    timer = Timer((args.N_iters - start) // args.i_testset)
    hist_psnr = hist_psnr1 = hist_psnr2 = n_pseudo_img = n_seen_img = 0
    global global_step
    print('Begin training')
    hard_pool_full = False
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    for i in trange(start+1, args.N_iters+1):
        t0 = time.time()
        global_step = i
        loss_line = LossLine()

        # update LR
        if args.lr: # use our own lr schedule
            new_lrate = lr_scheduler(optimizer, global_step)
        else:
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

        # Sample random ray batch
        if use_batching: # @mst: False in default
            # Random over all images
            batch = rays_rgb[i_batch: i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        
        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            
            # KD, update dataloader
            if args.datadir_kd:
                if args.data_mode in ['images']:
                    if i % args.i_update_data == 0: # update trainloader, possibly load more data
                        if args.dataset_type == 'blender':
                            t_ = time.time()
                            pr = get_pseudo_ratio(args.pseudo_ratio_schedule, i)
                            trainloader, n_total_img = get_dataloader(args.dataset_type, args.datadir_kd.split(':')[1], pseudo_ratio=pr)
                            print(f'Iter {i}. Reloaded data (time: {time.time()-t_:.2f}s). Now total #train images: {n_total_img} Pseudo_ratio: {pr:.4f}')

                    # get pose and target
                    if args.dataset_type == 'blender':
                        target, pose, img_i = [x[0] for x in trainloader.next()] # batch size = 1
                        target, pose = target.to(device), pose.to(device)
                        pose = pose[:3, :4]
                        if img_i >= n_original_img:
                            n_pseudo_img += 1
                    else: # LLFF dataset
                        use_pseudo_img = torch.rand(1) < len(kd_poses) / (len(train_poses) + len(kd_poses))
                        if use_pseudo_img:
                            img_i = np.random.permutation(len(kd_poses))[0]
                            pose = kd_poses[img_i, :3, :4]
                            target = kd_targets[img_i]
                            n_pseudo_img += 1
                    
                    n_seen_img += 1
                    loss_line.update('pseudo_img_ratio', n_pseudo_img/ n_seen_img, '.4f')
                
                elif args.data_mode in ['rays', '16x16patches_v2', '16x16patches_v3']:
                    if i % args.i_update_data == 0: # update trainloader, possibly load more data
                        if args.dataset_type == 'blender':
                            t_ = time.time()
                            trainloader, n_total_img = get_dataloader(args.dataset_type, args.datadir_kd.split(':')[1])
                            print(f'Iter {i}. Reloaded data (time: {time.time()-t_:.2f}s). Now total #train images: {n_total_img}')
                
            # get rays (rays_o, rays_d, target_s)
            if N_rand is not None:
                if args.data_mode in ['images']:
                    rays_o, rays_d = get_rays(H, W, focal, pose)  # (H, W, 3), (H, W, 3), origin: (-1.8393, -1.0503,  3.4298)
                    if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                            ), -1)
                        if i == start + 1:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                    # select pixels as a batch
                    select_coords, patch_bbx = get_selected_coords(coords, N_rand, args.select_pixel_mode)

                    # get rays_o and rays_d for the selected pixels
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    
                    # get target for the selected pixels
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                
                elif args.data_mode in ['rays']:
                    rays_o, rays_d, target_s = trainloader.next() # rays_o: [N_rand, 4096, 3] rays_d: [N_rand, 4096, DIM_DIR] target_s: [N_rand, 4096, DIM_RGB]
                    rays_o, rays_d, target_s = rays_o.to(device), rays_d.to(device), target_s.to(device)
                    rays_o = rays_o.view(-1, 3) # [N_rand*4096, 3]
                    rays_d = rays_d.view(-1, DIM_DIR) # [N_rand*4096, DIM_DIR]
                    target_s = target_s.view(-1, DIM_RGB) # [N_rand*4096, DIM_RGB]

                    if args.shuffle_input:
                        rays_d = rays_d.view(rays_d.shape[0], DIM_DIR//3, 3) # [N_rand*4096, DIM_DIR//3, 3]
                        target_s = target_s.view(target_s.shape[0], DIM_RGB//3, 3) # [N_rand*4096, DIM_RGB//3, 3]
                        shuffle_input_randix = torch.randperm(DIM_DIR//3)
                        rays_d = rays_d[:, shuffle_input_randix, :]
                        target_s = target_s[:, shuffle_input_randix, :]
                        rays_d = rays_d.view(-1, DIM_DIR) # [N_rand*4096, DIM_DIR] 
                        target_s = target_s.view(-1, DIM_RGB) # [N_rand*4096, DIM_RGB]
                    
                    if args.model_name in ['nerf_v4']: # in nerf_v4, load two rays each time
                        rays_d, rays_d2 = rays_d[:, :3], rays_d[:, 3:6] # [N_rand*4096, 3]
                        target_s, target_s2 = target_s[:, :3], target_s[:, 3:6] # [N_rand*4096, 3]
                
                elif args.data_mode in ['images_new']:
                    rays_o, rays_d, target_s = trainloader.next() # all shapes are: [N_rand, crop_size, crop_size, 3]
                    rays_o, rays_d, target_s = rays_o.to(device), rays_d.to(device), target_s.to(device)
                    target_s = target_s.permute(0, 3, 1, 2) # [N_rand, 3, crop_size, crop_size]
                
                elif args.data_mode in ['16x16patches']:
                    rays_o, rays_d, target_s = trainloader.next() # all shapes are: [N_rand, 16, 16, 3]
                    bbx1, bby1, bbx2, bby2 = square_rand_bbox(img_h=rays_o.shape[1], img_w=rays_o.shape[2], rand_crop_size=args.rand_crop_size)
                    rays_o, rays_d, target_s = rays_o[:, bby1:bby2, bbx1:bbx2, :], rays_d[:, bby1:bby2, bbx1:bbx2, :], target_s[:, bby1:bby2, bbx1:bbx2, :]
                    rays_o, rays_d, target_s = rays_o.to(device), rays_d.to(device), target_s.to(device) # all shapes are: [N_rand, 3, 3, 3]

                elif args.data_mode in ['16x16patches_v2']:
                    rays_o, rays_d, target_s = trainloader.next() # all shapes are: [N_rand, 16, 16, 3]
                    rays_o, rays_d, target_s = rays_o.to(device), rays_d.to(device), target_s.to(device)
                    target_s = target_s.permute(0, 3, 1, 2) # [N_rand, 3, 16, 16]

                elif args.data_mode in ['16x16patches_v3']:
                    rays_o, rays_d, target_s = trainloader.next() # all shapes are: [N_rand, 32, 16, 16, 3]
                    rays_o, rays_d, target_s = rays_o.to(device), rays_d.to(device), target_s.to(device)
                    rays_o = rays_o.view(-1, *rays_o.shape[-3:]) # [N_rand*32, 16, 16, 3]
                    rays_d = rays_d.view(-1, *rays_d.shape[-3:]) # [N_rand*32, 16, 16, 3]
                    target_s = target_s.view(-1, *target_s.shape[-3:]) # [N_rand*32, 16, 16, 3]
                    target_s = target_s.permute(0, 3, 1, 2) # [N_rand*32, 3, 16, 16]

            batch_size = rays_o.shape[0]
            if args.hard_ratio:
                if isinstance(args.hard_ratio, list):
                    n_hard_in = int(args.hard_ratio[0] * batch_size) # the number of hard samples into the hard pool
                    n_hard_out = int(args.hard_ratio[1] * batch_size) # the number of hard samples out of the hard pool
                else:
                    n_hard_in = int(args.hard_ratio * batch_size)
                    n_hard_out = n_hard_in
                n_hard_in = min(n_hard_in, n_hard_out) # n_hard_in <= n_hard_out

            if hard_pool_full:
                rand_ix_out = np.random.permutation(hard_rays.shape[0])[:n_hard_out]
                picked_hard_rays = hard_rays[rand_ix_out]
                rays_o   = torch.cat([rays_o,   picked_hard_rays[:,  :3]], dim=0)
                rays_d   = torch.cat([rays_d,   picked_hard_rays[:, 3:6]], dim=0)
                target_s = torch.cat([target_s, picked_hard_rays[:, 6: ]], dim=0)
            
        # update data time
        data_time.update(time.time() - t0)

        # forward and get loss
        loss = 0
        if args.model_name == 'nerf':
            rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_train)
            if 'rgb0' in extras:
                loss += img2mse(extras['rgb0'], target_s)

        elif args.model_name in ['nerf_v2']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            if args.directly_predict_rgb:
                rgb, *_ = model(rays_o, rays_d, global_step=global_step, perturb=perturb)
            else:
                rgb, *_, raw, pts, viewdirs = model(rays_o, rays_d, global_step=global_step, perturb=perturb)
        
        elif args.model_name in ['nerf_v3']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            rgb, *_ = model(rays_o, rays_d, global_step=global_step, perturb=perturb)
        
        elif args.model_name in ['nerf_v3.2']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            pts = point_sampler.sample_train(rays_o, rays_d, perturb=perturb)
            rgb = model(positional_embedder(pts))

        elif args.model_name in ['nerf_v3.3']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            pts = point_sampler.sample_train(rays_o, rays_d, perturb=perturb)
            rgb = model.forward_mlp(positional_embedder(pts))
        
        elif args.model_name in ['nerf_v3.4']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            pts = point_sampler.sample_train2(rays_o, rays_d, perturb=perturb)
            pts = positional_embedder.embed(pts) # [n_img, patch_h, patch_w, n_sample, 3, 2L+1]
            rgb = model(pts.view(pts.shape[0], -1))
            rgb = rgb.view(*pts.shape[:3], 3) # [n_img, patch_h, patch_w, 3]

        elif args.model_name in ['nerf_v3.4.2', 'nerf_v3.6', 'nerf_v3.7']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            rays_o = rays_o.repeat(1, args.scale**2) # [n_ray, 3] -> [n_ray, 27], to match the shape of rays_d, rays_rgb
            pts = point_sampler.sample_train(rays_o, rays_d, perturb=perturb) # [n_ray, n_sample * DIM_DIR]
            pts = positional_embedder(pts)
            rgb = model(pts) # [n_ray, 3 * scale ** 2]

        elif args.model_name in ['nerf_v3.8']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            pts = point_sampler.sample_train_cnnstyle(rays_o, rays_d, perturb=perturb) # [n_img, patch_h, patch_w, n_sample, 3]
            pts = positional_embedder.embed_cnnstyle(pts) # [n_img, patch_h, patch_w, n_sample, 3, 2L+1]
            pts = pts.view(*pts.shape[:3], -1) # [n_img, patch_h, patch_w, n_sample*3*2L+1]
            pts = pts.permute(0, 3, 1, 2) # [n_img, n_sample*3*2L+1, patch_h, patch_w]
            rgb = model(pts) # [n_img, 3, patch_h, patch_w]

        elif args.model_name in ['nerf_v3.5']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            if args.rand_crop_size > 0:
                img_h = img_w = args.rand_crop_size
            else:
                img_h, img_w = IMG_H, IMG_W
            pts = point_sampler.sample_train(rays_o, rays_d, perturb=perturb)
            rgb = model.forward_mlp2(positional_embedder(pts), img_h=img_h, img_w=img_w)

        elif args.model_name in ['nerf_v4']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            rgb, rgb2 = model(rays_o, rays_d, rays_d2, scale=args.forward_scale, perturb=perturb)
        
        elif args.model_name in ['nerf_v6']:
            model = render_kwargs_train['network_fn']
            perturb = render_kwargs_train['perturb']
            shape = rays_o.shape # [N_rand, crop_size, crop_size, 3]
            pts = point_sampler.sample_train(rays_o.view(-1, 3), rays_d.view(-1, 3), perturb=perturb)
            pts = positional_embedder(pts) # [N_rand*crop_size*crop_size, embed_dim]
            pts = pts.view(shape[0], shape[1], shape[2], -1) # [N_rand, crop_size, crop_size, embed_dim]
            pts = pts.permute(0, 3, 1, 2) # [N_rand, embed_dim, crop_size, crop_size]
            rgb = model(pts) # [N_rand, 3, crop_size, crop_size]
        
        # rgb loss
        loss_rgb = img2mse(rgb, target_s)
        psnr = mse2psnr(loss_rgb)
        loss_line.update('psnr', psnr.item(), '.4f')
        if not (args.enhance_cnn and args.freeze_pretrained):
            loss += loss_rgb
        
        if args.model_name in ['nerf_v4']:
            loss_rgb2 = img2mse(rgb2, target_s2)
            psnr2 = mse2psnr(loss_rgb2)
            loss_line.update('psnr2', psnr2.item(), '.4f')
            loss += loss_rgb2

        # enhance cnn rgb loss
        if args.enhance_cnn:
            model_enhance = render_kwargs_train['network_enhance']
            rgb1 = model_enhance(rgb)
            loss_rgb1 = img2mse(rgb1, target_s)
            psnr1 = mse2psnr(loss_rgb1)
            loss_line.update('psnr1', psnr1.item(), '.4f')
            loss += loss_rgb1
        
        # backward and update
        optimizer.zero_grad()
        loss.backward()

        # group l2 regularization
        if args.group_l2:
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear)):
                    norm = torch.norm(m.weight.data, p=2, dim=-1, keepdim=True) # [d_out, 1]
                    grad = m.weight.data / norm.expand_as(m.weight.data) # [d_out, d_in]
                    m.weight.grad.data.add_(args.group_l2 * grad)
        
        optimizer.step()
        batch_time.update(time.time() - t0)

        # collect hard examples
        if args.hard_ratio:
            _, indices = torch.sort( torch.mean((rgb[:batch_size] - target_s[:batch_size]) ** 2, dim=1) )
            hard_indices = indices[-n_hard_in:]
            hard_rays_ = torch.cat([rays_o[hard_indices], rays_d[hard_indices], target_s[hard_indices]], dim=-1)
            if hard_pool_full:
                hard_rays[rand_ix_out[:n_hard_in]] = hard_rays_ # replace
            else:
                hard_rays = torch.cat([hard_rays, hard_rays_], dim=0) # append
                if hard_rays.shape[0] >= batch_size * args.hard_mul:
                    hard_pool_full = True

        # smoothing for log print
        if not math.isinf(psnr.item()):
            hist_psnr = psnr.item() if i == start + 1 else hist_psnr * 0.95 + psnr.item() * 0.05
            loss_line.update('hist_psnr', hist_psnr, '.4f')
        if args.model_name in ['nerf_v4'] and not math.isinf(psnr2.item()):
            hist_psnr2 = psnr2.item() if i == start + 1 else hist_psnr2 * 0.95 + psnr2.item() * 0.05
            loss_line.update('hist_psnr2', hist_psnr2, '.4f')
        if args.enhance_cnn:
            hist_psnr1 = psnr1.item() if i == start + 1 else hist_psnr1 * 0.95 + psnr1.item() * 0.05
            loss_line.update('hist_psnr1', hist_psnr1, '.4f')

        # print logs of training
        if i % args.i_print == 0:
            logstr = f"[TRAIN] Iter {i} data_time {data_time.val:.4f} ({data_time.avg:.4f}) batch_time {batch_time.val:.4f} ({batch_time.avg:.4f}) " + loss_line.format()
            print(logstr)

            # save image for check
            if args.enhance_cnn and i % (100 * args.i_print) == 0:
                img = rgb1.reshape([patch_h, patch_w, 3])
                save_path = f'{logger.gen_img_path}/train_patch_{ExpID}_iter{i}.png'
                imageio.imwrite(save_path, to8b(img))

        # check gradients to make sure group_l2 works normally
        if args.group_l2 and i % (args.i_print * 10) == 0:
            n_neuron_print = 10
            print(f'iter {i} neuron norms (penalty factor {args.group_l2}):')
            for name, m in model.named_modules():
                if isinstance(m, (nn.Linear)):
                    logstr = ['%.6f' % x for x in torch.norm(m.weight.data, p=2, dim=-1)[:n_neuron_print]]
                    logstr = f'{name:<22s} ' + ' '.join(logstr)
                    netprint(logstr)
        
        # test: using the splitted test images
        if i % args.i_testset == 0:
            testsavedir = f'{logger.gen_img_path}/testset_{ExpID}_iter{i}' # save the renderred test images
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                print(f'Iter {i} Testing...')
                t_ = time.time()
                *_, test_loss, test_psnr, errors = render_path(test_poses, hwf, args.chunk, render_kwargs_test, gt_imgs=test_images, 
                    savedir=testsavedir, render_factor=args.render_factor)
                t_test = time.time() - t_
            accprint(f'[TEST] Iter {i} Loss {test_loss.item():.4f} PSNR {test_psnr.item():.4f} Train_HistPSNR {hist_psnr:.4f} LR {new_lrate:.8f} Time {t_test:.1f}s')
            print(f'Saved rendered test images: "{testsavedir}"')
            print(f'Predicted finish time: {timer()}')

        # test: using novel poses
        if i % args.i_video == 0:
            with torch.no_grad():
                print(f'Iter {i} Rendering video... (n_pose: {len(video_poses)})')
                t_ = time.time()
                rgbs, disps, *_, video_loss, video_psnr, errors = render_path(video_poses, hwf, args.chunk, render_kwargs_test, 
                        gt_imgs=video_targets, render_factor=args.render_factor)
                t_video = time.time() - t_
            video_path = f'{logger.gen_img_path}/video_{ExpID}_iter{i}_{args.video_tag}.mp4'
            imageio.mimwrite(video_path, to8b(rgbs), fps=30, quality=8)
            # imageio.mimwrite(disp_path, to8b(disps / np.max(disps)), fps=30, quality=8)
            print(f'[VIDEO] Rendering done. Time {t_video:.2f}s. Save video: "{video_path}"')
            if video_psnr is not None:
                print(f'[VIDEO] video_loss {video_loss.item():.4f} video_psnr {video_psnr.item():.4f}')
                imageio.mimwrite(video_path.replace('.mp4', '_error.mp4'), to8b(errors), fps=30, quality=8)

        # save checkpoint
        if i % args.i_weights == 0:
            path = os.path.join(logger.weights_path, 'ckpt.tar'.format(i))
            to_save = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if args.model_name in ['nerf'] and args.N_importance > 0:
                to_save['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            if args.model_name in ['nerf_v3.2']:
                to_save['network_fn'] = render_kwargs_train['network_fn']
            if args.enhance_cnn:
                to_save['network_enhance'] = render_kwargs_train['network_enhance'],
                to_save['network_enhance_state_dict'] = render_kwargs_train['network_enhance'].state_dict()
            torch.save(to_save, path)
            save_log = f'Iter {i} Save checkpoint: "{path}"'
            
            # # convert to onnx
            # onnx_path = path.replace('.tar', '.onnx')
            # save_onnx(model, onnx_path)
            # save_log += f', onnx saved at "{onnx_path}"'
            print(save_log)

if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
