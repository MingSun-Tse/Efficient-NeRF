import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import NeRF, sample_pdf
from run_nerf_helpers_v2 import NeRF_v2, img2mse, mse2psnr, to8b, get_rays, get_embedder, get_novel_poses, get_novel_poses_v2

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, setup_blender_datadir, save_blender_data
from collections import OrderedDict
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

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


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, new_render_func=False):
    H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    
    rgbs, disps = [], []
    for i, c2w in enumerate(render_poses):
        if new_render_func: # our new rendering func
            model = render_kwargs['network_fn']
            perturb = render_kwargs['perturb']
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(c2w))
            rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
            rgb, disp = [], []
            for ix in range(0, rays_o.shape[0], chunk):
                rgb_, disp_, *_ = model(rays_o[ix: ix+chunk], rays_d[ix: ix+chunk], perturb=perturb)
                rgb += [rgb_]
                disp += [disp_]
            rgb, disp = torch.cat(rgb, dim=0), torch.cat(disp, dim=0)
            rgb, disp = rgb.view(H, W, -1), disp.view(H, W, -1)
        else: # original implementation
            rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    
    if gt_imgs is not None:
        rgbs = torch.from_numpy(rgbs).to(device)
        test_loss = img2mse(rgbs, gt_imgs)
        test_psnr = mse2psnr(test_loss)
        rgbs = rgbs.data.cpu().numpy() # change back to np array type
    else:
        test_loss, test_psnr = None, None

    return rgbs, disps, test_loss, test_psnr

def _load_weights(model, ckpt_path, key):
    ckpt_path = check_path(ckpt_path)
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt[key]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return ckpt_path, ckpt

def create_nerf(args, near, far):
    """Instantiate NeRF's MLP model.
    """
    # set up model
    model_fine = network_query_fn = None
    if args.model_name == 'nerf':
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        input_ch_views = 0
        embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]
        model = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars = list(model.parameters())

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
        grad_vars = list(model.parameters())
    
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
        ckpt_path, ckpt = _load_weights(teacher_fn, args.teacher_ckpt, 'network_fn_state_dict')
        ckpt_path, ckpt = _load_weights(teacher_fine, args.teacher_ckpt, 'network_fine_state_dict')
        print(f'Load teacher ckpt successfully: "{ckpt_path}"')

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
        ckpt_path, ckpt = _load_weights(model, args.pretrained_ckpt, 'network_fn_state_dict')
        if model_fine is not None:
            _load_weights(model_fine, args.pretrained_ckpt, 'network_fine_state_dict')
        print('Load pretrained ckpt successfully: "%s"' % ckpt_path)
        
        # resume optimizer and iteration number if necessary
        if args.resume:
            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('Resume optimizer successfully')

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
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    if args.teacher_ckpt:
        render_kwargs_train['teacher_fn'] = teacher_fn
        render_kwargs_train['teacher_fine'] = teacher_fine

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
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    # @mst: 1e10 for infinite distance

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1) # @mst: direction vector needs normalization. why this * ?

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3], RGB for each sampled point 
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]

    # print to check alpha
    if verbose and global_step % args.i_print == 0:
        for i_ray in range(0, alpha.shape[0], 100):
            logtmp = ['%.4f' % x  for x in alpha[i_ray]]
            netprint('%4d: ' % i_ray + ' '.join(logtmp))

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] # @mst: [N_rays, N_samples]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
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
      ray_batch: array of shape [batch_size, ...]. All information necessary
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

    t_vals = torch.linspace(0., 1., steps=N_samples)
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
        t_rand = torch.rand(z_vals.shape) # uniform dist [0, 1)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

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
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

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

# set up logging directories -------
from logger import Logger
from utils import Timer, check_path, LossLine, PresetLRScheduler, strdict_to_dict
from option import args

logger = Logger(args)
print = logger.log_printer.logprint
accprint = logger.log_printer.accprint
netprint = logger.log_printer.netprint
ExpID = logger.ExpID
# ---------------------------------

def get_teacher_target(poses, H, W, focal, render_kwargs_train, args):
    render_kwargs_ = {x: v for x, v in render_kwargs_train.items()}
    render_kwargs_['network_fn'] = render_kwargs_train['teacher_fn'] # temporarily change the network_fn
    render_kwargs_['network_fine'] = render_kwargs_train['teacher_fine'] # temporarily change the network_fine
    render_kwargs_.pop('teacher_fn')
    render_kwargs_.pop('teacher_fine')
    teacher_target = []
    t_ = time.time()
    for ix, pose in enumerate(poses):
        print(f'[{ix}/{len(poses)}] Using teacher to render more images...')
        rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))
        batch_rays = torch.stack([rays_o, rays_d], 0)
        rgb, *_ = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                        verbose=False, retraw=False,
                                        **render_kwargs_)
        teacher_target.append(rgb)
    print(f'Teacher rendering done ({len(poses)} poses). Time: {(time.time() - t_):.2f}s')
    return teacher_target

def train():
    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

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
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip, n_pose=args.n_pose_video)
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

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    test_poses = torch.Tensor(poses[i_test]).to(device)
    test_images = torch.Tensor(images[i_test]).to(device)

    # Create log dir and copy the config file
    basedir = logger.exp_path # args.basedir @mst: use our experiment path
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, near, far)

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.test_teacher:
        assert args.teacher_ckpt
        print('Testing teacher...')
        render_kwargs_ = {x: v for x, v in render_kwargs_test.items()}
        render_kwargs_['network_fn'] = render_kwargs_train['teacher_fn'] # temporarily change the network_fn
        render_kwargs_['network_fine'] = render_kwargs_train['teacher_fine'] # temporarily change the network_fine
        with torch.no_grad():
            *_, test_loss, test_psnr = render_path(test_poses, hwf, 4096, render_kwargs_, gt_imgs=test_images, render_factor=args.render_factor, new_render_func=False)
        print(f'Teacher test: Loss {test_loss.item():.4f} PSNR {test_psnr.item():.4f}')

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    train_images, train_poses = images[i_train], poses[i_train]

    # --- generate new data using trained NeRF
    datadir_kd_old, datadir_kd_new = args.datadir_kd.split(':')
    setup_blender_datadir(datadir_kd_old, datadir_kd_new)
    print('Set up new data directory, done!')

    # get poses of psuedo data
    kd_poses = get_novel_poses_v2(args, n_pose=args.n_pose_kd).to(device)
    n_new_pose = len(kd_poses)
    rand_ix = np.random.permutation(n_new_pose)
    kd_poses = kd_poses[rand_ix] # shuffle
    print('Get new poses, done!')

    # render pseduo data
    batch_size = 100
    n_img = len(train_images)
    for i in range(0, n_new_pose, batch_size):
        poses = kd_poses[i: i+batch_size]
        n_img += len(poses)
        teacher_target = get_teacher_target(poses, H, W, focal, render_kwargs_train, args)
        if args.dataset_type == 'blender':
            save_blender_data(datadir_kd_new, kd_poses, teacher_target)
        print(f'Create new data. Save to "{datadir_kd_new}". Now total #train samples: {n_img}')
    # ---
    
if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
