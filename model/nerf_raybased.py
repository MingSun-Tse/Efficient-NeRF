from os import getgroups
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np, time, math
import dill

# TODO: remove this dependency
# from torchsearchsorted import searchsorted # not used by raybased nerf, commented because it is not compiled successfully using docker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Misc
to_tensor = lambda x: x.to(device) if isinstance(x, torch.Tensor) else torch.Tensor(x).to(device)
to_array = lambda x: x if isinstance(x, np.ndarray) else x.data.cpu().numpy()
to_list = lambda x: x if isinstance(x, list) else to_array(x).tolist()
to8b = lambda x : (255 * np.clip(to_array(x), 0, 1)).astype(np.uint8)
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(to_tensor([10.]))

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim # @mst: (1) for x: 63 = (2x10+1)x3. 10 from the paper. 1 because of the 'include_input = True' (2) for view, 27 = (2x4+1)x3
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class PointSampler():
    def __init__(self, H, W, focal, n_sample, near, far):
        self.H, self.W = H, W
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
        i, j = i.t(), j.t()
        self.dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], dim=-1).to(device) # [H, W, 3]
        
        t_vals = torch.linspace(0., 1., steps=n_sample).to(device) # [n_sample]
        self.z_vals = near * (1 - t_vals) + far * (t_vals) # [n_sample]
        self.z_vals_test = self.z_vals[None, :].expand(H*W, n_sample) # [H*W, n_sample]

    def sample_test(self, c2w): # c2w: [3, 4]
        rays_d = torch.sum(self.dirs.unsqueeze(dim=-2) * c2w[:3,:3], dim=-1).view(-1, 3) # [H*W, 3] # TODO-@mst: improve this non-intuitive impl.
        rays_o = c2w[:3, -1].expand(rays_d.shape) # [H*W, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * self.z_vals_test[..., :, None] # [H*W, n_sample, 3]
        return pts.view(pts.shape[0], -1) # [H*W, n_sample*3]
    
    def sample_test2(self, c2w): # c2w: [3, 4]
        rays_d = torch.sum(self.dirs.unsqueeze(dim=-2) * c2w[:3,:3], dim=-1).view(-1, 3) # [H*W, 3] # TODO-@mst: improve this non-intuitive impl.
        rays_o = c2w[:3, -1].expand(rays_d.shape) # [H*W, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * self.z_vals_test[..., :, None] # [H*W, n_sample, 3]
        return pts # [..., n_sample, 3]

    def sample_train(self, rays_o, rays_d, perturb):
        z_vals = self.z_vals[None, :].expand(rays_o.shape[0], self.z_vals.shape[0]) # depth [n_ray, n_sample]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1],  mids], dim=-1)
            t_rand = torch.rand(z_vals.shape).to(device) # [n_ray, n_sample]
            z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [n_ray, n_sample, DIM_DIR]
        return pts.view(pts.shape[0], -1) # [n_ray, n_sample * DIM_DIR]

    def sample_train2(self, rays_o, rays_d, perturb): # rays_o: [n_img, patch_h, patch_w, 3]
        z_vals = self.z_vals[None, None, None, :].expand(*rays_o.shape[:3], self.z_vals.shape[0]) # [n_img, patch_h, patch_w, n_sample]
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1]) # [n_img, patch_h, patch_w, n_sample]
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1],  mids], dim=-1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape[0]).to(device) # [n_img]
            t_rand = t_rand[:, None, None, None].expand_as(z_vals)  # [n_img, patch_h, patch_w, n_sample]
            z_vals = lower + (upper - lower) * t_rand  # [n_img, patch_h, patch_w, n_sample]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [n_img, patch_h, patch_w, n_sample, 3]
        return pts

class PositionalEmbedder():
    def __init__(self, L, include_input=True):
        self.weights = 2 ** torch.linspace(0, L-1, steps=L).to(device) # [L]
        self.include_input = include_input
        self.embed_dim = 2 * L + 1 if include_input else 2 * L
    def __call__(self, x): 
        y = x[..., None] * self.weights # [n_ray, dim_pts, 1] * [L] -> [n_ray, dim_pts, L]
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1) # [n_ray, dim_pts, 2L]
        if self.include_input:
            y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1) # [n_ray, dim_pts, 2L+1]
        return y.view(y.shape[0], -1) # [n_ray, dim_pts*(2L+1)], example: 48*21=1008
    def embed(self, x): # v2 of __call__
        y = x[..., :, None] * self.weights 
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1) 
        if self.include_input:
            y = torch.cat([y, x.unsqueeze(dim=-1)], dim=-1)
        return y # [n_img, patch_h, patch_w, n_sample, 3, 2L+1]

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, global_step=-1, print=print):
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
    if global_step % 100 == 0:
        for i_ray in range(0, alpha.shape[0], 100):
            logtmp = ['%.4f' % x  for x in alpha[i_ray]]
            print('%4d: ' % i_ray + ' '.join(logtmp))

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] # @mst: [N_rays, N_samples]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

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

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

class NeRF_v2(nn.Module):
    '''Raybased nerf. Similar arch to NeRF (using skip layers and view direction as input)'''
    def __init__(self, args, near, far, print=print):
        super(NeRF_v2, self).__init__()
        self.args = args
        self.near = near
        self.far = far
        D, W = args.netdepth, args.netwidth
        self.skips = [int(x) for x in args.skips.split(',')] if args.skips else []
        self.print = print
        assert args.n_sample_per_ray >= 1
        n_sample = args.n_sample_per_ray

        # positional embedding function
        self.embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        if args.use_viewdirs:
            self.embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

        # body network
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch * n_sample, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch * n_sample, W) for i in range(D - 1)])
            
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views * n_sample + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if args.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            if not args.directly_predict_rgb:
                self.alpha_linear = nn.Linear(W, n_sample)
            self.rgb_linear = nn.Linear(W//2, 3 * n_sample)
        else:
            output_ch = 5 if args.N_importance > 0 else 4
            self.output_linear = nn.Linear(W, output_ch)

        # original NeRF forward impl.
        self.network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=self.embed_fn,
                                                                    embeddirs_fn=self.embeddirs_fn,
                                                                    netchunk=args.netchunk)

    def forward(self, rays_o, rays_d, global_step=-1, perturb=0):
        n_ray = rays_o.size(0)
        n_sample = self.args.n_sample_per_ray

        if n_sample > 1:
            t_vals = torch.linspace(0., 1., steps=n_sample).to(device)
            t_vals = t_vals[None, :].expand(n_ray, n_sample)
            z_vals = self.near * (1 - t_vals) + self.far * (t_vals)
            if perturb > 0.:
                # get intervals between samples
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(device) # uniform dist [0, 1)
                z_vals = lower + (upper - lower) * t_rand
        
            # get sample coordinates
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [n_ray, n_sample, 3]

            # positional embedding
            embedded_pts = self.embed_fn(pts.view(-1, 3)) # shape: [n_ray*n_sample_per_ray, 63]
            embedded_pts = embedded_pts.view(rays_o.size(0), -1) # [n_ray, n_sample_per_ray*63]

            # pose embedding
            if self.args.use_viewdirs:
                # provide ray directions as input
                viewdirs = rays_d
                # if c2w_staticcam is not None:
                #     # special case to visualize effect of viewdirs
                #     rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
                viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) # @mst: 'rays_d' is real-world data, needs normalization.
                viewdirs = torch.reshape(viewdirs, [-1, 3]).float() # [n_ray, 3]
                dirs = viewdirs[:, None].expand(pts.shape) # [n_ray, 3] -> [n_ray, n_sample, 3]

                if perm_invar: # permutation invariance prior
                    dirs = dirs[:, rand_index]

                dirs = torch.reshape(dirs, (-1, 3))
                embedded_dirs = self.embeddirs_fn(dirs)
                embedded_dirs = embedded_dirs.view(rays_o.size(0), -1) # [n_ray, n_sample * 27]
        
        else: # n_sample = 1
            embedded_pts = self.embed_fn(rays_o) # [n_ray, 63]
            embedded_dirs = self.embeddirs_fn(rays_d / rays_d.norm(dim=-1, keepdim=True)) # [n_ray, 27]

        # body network
        h = embedded_pts
        for i, layer in enumerate(self.pts_linears):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([embedded_pts, h], dim=-1)

        # get raw outputs
        if self.args.use_viewdirs:
            # get rgb
            feature = self.feature_linear(h)
            h = torch.cat([feature, embedded_dirs], -1)
            for i, layer in enumerate(self.views_linears):
                h = F.relu(layer(h))
            rgb = self.rgb_linear(h)

            # get alpha
            if self.args.directly_predict_rgb:
                raw = rgb # # [n_ray, n_sample * 3]
            else:
                alpha = self.alpha_linear(h) # [n_ray, n_sample]
                raw = torch.cat([rgb, alpha], dim=-1) # [n_ray, n_sample * 4]
            raw = raw.view(n_ray, n_sample, -1) # [n_ray, n_sample, -1]
        
        # rendering
        if self.args.directly_predict_rgb:
            rgb_map = torch.sigmoid(raw[..., :3].mean(dim=1)) # [n_ray, 3]
            disp_map = rgb_map # placeholder
            return rgb_map, disp_map
        else: # use rendering equation
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, self.args.raw_noise_std, white_bkgd=False, pytest=False, global_step=global_step, print=self.print)
            return rgb_map, disp_map, acc_map, weights, depth_map, raw, pts, viewdirs

class NeRF_v3(nn.Module):
    '''No input skip layers. Move view direction as input to the very 1st layer.'''
    def __init__(self, args, near, far):
        super(NeRF_v3, self).__init__()
        self.args = args
        self.near = near
        self.far = far
        assert args.n_sample_per_ray >= 1
        n_sample = args.n_sample_per_ray
        D, W = args.netdepth, args.netwidth

        # positional embedding function
        self.embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        if args.use_viewdirs:
            self.embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

        # network
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D - 1) + [3]
        
        input_dim = input_ch + input_ch_views if args.use_viewdirs else input_ch 
        input_dim *= n_sample

        # head
        self.head = nn.Sequential(*[nn.Linear(input_dim, Ws[0]), nn.ReLU(inplace=True)])
        input_dim = Ws[0]
        
        # body
        body = []
        for i in range(1, D - 1):
            body += [nn.Linear(input_dim, Ws[i]), nn.ReLU(inplace=True)]
            input_dim = Ws[i]
        self.body = nn.Sequential(*body)
        
        # tail
        if args.linear_tail:
            self.tail = nn.Linear(input_dim, 3)
        else:
            self.tail = nn.Sequential(*[nn.Linear(input_dim, 3), nn.Sigmoid()])
        
        self.t_vals = torch.linspace(0., 1., steps=n_sample).to(device)

    def forward(self, rays_o, rays_d, global_step=-1, perturb=0):
        n_ray = rays_o.size(0)
        n_sample = self.args.n_sample_per_ray

        t0 = time.time()
        if n_sample > 1:
            t_vals = self.t_vals
            t_vals = t_vals[None, :].expand(n_ray, n_sample)
            z_vals = self.near * (1 - t_vals) + self.far * (t_vals)
            if perturb > 0.:
                # get intervals between samples
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(device) # uniform dist [0, 1)
                z_vals = lower + (upper - lower) * t_rand

            # get sample coordinates
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [n_ray, n_sample, 3]

            # positional embedding
            embedded_pts = self.embed_fn(pts.view(-1, 3)) # shape: [n_ray*n_sample_per_ray, 63]
            embedded_pts = embedded_pts.view(rays_o.size(0), -1) # [n_ray, n_sample_per_ray*63]

            # pose embedding
            if self.args.use_viewdirs:
                # provide ray directions as input
                viewdirs = rays_d
                # if c2w_staticcam is not None:
                #     # special case to visualize effect of viewdirs
                #     rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
                viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) # @mst: 'rays_d' is real-world data, needs normalization.
                viewdirs = torch.reshape(viewdirs, [-1, 3]).float() # [n_ray, 3]
                dirs = viewdirs[:, None].expand(pts.shape) # [n_ray, 3] -> [n_ray, n_sample, 3]
                dirs = torch.reshape(dirs, (-1, 3)) # [n_ray * n_sample, 3]
                embedded_dirs = self.embeddirs_fn(dirs) # [n_ray * n_sample, 27]
                embedded_dirs = embedded_dirs.view(rays_o.size(0), -1) # [n_ray, n_sample * 27]
                embedded_pts = torch.cat([embedded_pts, embedded_dirs], dim=-1)

        else: # n_sample = 1
            embedded_pts = self.embed_fn(rays_o) # [n_ray, 63]
            if self.args.use_viewdirs:
                embedded_dirs = self.embeddirs_fn(rays_d / rays_d.norm(dim=-1, keepdim=True)) # [n_ray, 27]
                embedded_pts = torch.cat([embedded_pts, embedded_dirs], dim=-1)

        # network forward
        h = self.head(embedded_pts)
        h = self.body(h) + h if self.args.use_residual else self.body(h)
        rgb = self.tail(h)
        return rgb, rgb
    
    def forward_(self, embedded_pts):
        '''move positional encoding out'''
        h = self.head(embedded_pts)
        h = self.body(h) + h if self.args.use_residual else self.body(h)
        rgb = self.tail(h)
        return rgb, rgb

class NeRF_v3_2(nn.Module):
    '''Based on NeRF_v3, move positional embedding out'''
    def __init__(self, args, input_dim):
        super(NeRF_v3_2, self).__init__()
        self.args = args
        D, W = args.netdepth, args.netwidth

        # get network width
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D-1) + [3]

        # head
        self.input_dim = input_dim
        self.head = nn.Sequential(*[nn.Linear(input_dim, Ws[0]), nn.ReLU(inplace=True)])
        
        # body
        body = []
        for i in range(1, D-1):
            body += [nn.Linear(Ws[i-1], Ws[i]), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        
        # tail
        self.tail = nn.Linear(input_dim, 3) if args.linear_tail else nn.Sequential(*[nn.Linear(Ws[D-2], 3), nn.Sigmoid()])
    
    def forward(self, x): # x: embedded position coordinates
        x = self.head(x)
        x = self.body(x) + x if self.args.use_residual else self.body(x)
        return self.tail(x)

class NeRF_v3_3(nn.Module):
    '''Use 1x1 conv to implement NeRF_v3_2. Keep the input data the same as that of NeRF_v3_2.'''
    def __init__(self, args, input_dim):
        super(NeRF_v3_3, self).__init__()
        self.args = args
        D, W = args.netdepth, args.netwidth

        # network width
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D - 1) + [3]
        
        # head
        self.input_dim = input_dim
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=input_dim, out_channels=Ws[0], kernel_size=1), nn.ReLU(inplace=True)])
        
        # body
        body = []
        for i in range(1, D - 1):
            body += [nn.Conv2d(in_channels=Ws[i-1], out_channels=Ws[i], kernel_size=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        
        # tail
        self.tail = nn.Conv2d(in_channels=Ws[i], out_channels=3, kernel_size=1) if args.linear_tail \
            else nn.Sequential(*[nn.Conv2d(in_channels=Ws[i], out_channels=3, kernel_size=1), nn.Sigmoid()])
        
    def forward(self, x): # x: [n_img, embed_dim, H, W]
        x = self.head(x)
        x = self.body(x) + x if self.args.use_residual else self.body(x)
        return self.tail(x) # [n_img, 3, H, W]

    def forward_mlp(self, x): # x: embedded position coordinates, [n_ray, input_dim]
        '''The input data format is for MLP network. To keep the data preparation the same as before.
        '''
        x = x.permute(1, 0) # [input_dim, n_ray]
        x = x.view(1, x.shape[0], x.shape[1]//32, -1) # [1, input_dim, H, W]
        x = self.forward(x)
        x = x.view(3, -1).permute(1, 0) # [n_ray, 3]
        return x

class NeRF_v3_4(nn.Module):
    '''Based on NeRF_v3.2, 3x3 rays share one network'''
    def __init__(self, args, input_dim, scale=3):
        super(NeRF_v3_4, self).__init__()
        self.args = args
        self.scale = scale
        D, W = args.netdepth, args.netwidth

        # get network width
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D-1) + [3]

        # head
        self.input_dim = input_dim * scale ** 2
        self.head = nn.Sequential(*[nn.Linear(self.input_dim, Ws[0]), nn.ReLU(inplace=True)])
        
        # body
        body = []
        for i in range(1, D-1):
            body += [nn.Linear(Ws[i-1], Ws[i]), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        
        # tail
        self.tail = nn.Linear(Ws[D-2], 3 * scale ** 2) if args.linear_tail \
            else nn.Sequential(*[nn.Linear(Ws[D-2], 3 * scale ** 2), nn.Sigmoid()])
    
    def forward(self, x): # x: embedded position coordinates
        x = self.head(x)
        x = self.body(x) + x if self.args.use_residual else self.body(x)
        return self.tail(x)

class Upsampler(nn.Sequential):
    '''refers to https://github.com/yulunzhang/RCAN/blob/3339ebc59519c3bb2b5719b87dd36515ec7f3ba7/RCAN_TrainCode/code/model/common.py#L58
    '''
    def __init__(self, conv, scale, n_feat, kernel_size=3, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, kernel_size, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, kernel_size, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class NeRF_v3_5(nn.Module):
    '''Based on NeRF_v3.2, 3x3 rays share one network. Use Conv and ShufflePixel'''
    def __init__(self, args, input_dim, scale=3):
        super(NeRF_v3_5, self).__init__()
        self.args = args
        D, W = args.netdepth, args.netwidth

        # get network width
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D-1) + [3]

        # head: downsize by scale
        self.input_dim = input_dim
        self.head = nn.Sequential(*[nn.Conv2d(input_dim, Ws[0], kernel_size=scale, stride=scale), nn.ReLU(inplace=True)])
        
        # body (keep the feature map size), use FC layers, a little bit faster than 1x1 conv
        body = []
        for i in range(1, D-1):
            body += [nn.Linear(Ws[i-1], Ws[i]), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)

        # tail: upsample by scale
        # --- use PixelShuffle, slow, deprecated!
        # conv = lambda in_channels, out_channels, kernel_size, bias: nn.Conv2d(in_channels, out_channels, kernel_size,
        #     padding=(kernel_size//2), bias=bias)
        # act = False if args.linear_tail else nn.Sigmoid
        # self.tail = nn.Sequential(*[
        #     Upsampler(conv, scale, Ws[i], kernel_size=1, act=act),
        #     conv(Ws[i], 3, kernel_size=1, bias=True)
        # ])
        # ---
        tail = [nn.Linear(Ws[i], 3 * scale**2)]
        if not args.linear_tail: tail.append(nn.Sigmoid())
        self.tail = nn.Sequential(*tail)

    # def forward(self, x): # x: embedded position coordinates
    #     '''if body is nn.Linear'''
    #     x = self.head(x) # [1, dim, H//scale, W//scale]
    #     shape = x.shape
    #     x = x.permute(0, 2, 3, 1) # [1, H//scale, W//scale, dim]
    #     x = x.view(-1, shape[1])
    #     x = self.body(x) # [1*H//scale*W//scale, dim]
    #     x = x.permute(1, 0)
    #     x = x.view(shape) # [1, dim, H//scale, W//scale]
    #     return self.tail(x) # [1, 3, H, W]
    
    def forward(self, x):
        '''if body is 1x1 conv'''
        torch.cuda.synchronize()
        t0 = time.time()
        x = self.head(x) 
        torch.cuda.synchronize()
        print(f'after head: {time.time() - t0:.6f}s')
        x = self.body(x) 
        torch.cuda.synchronize()
        print(f'after body: {time.time() - t0:.6f}s')
        x = self.tail(x)
        torch.cuda.synchronize()
        print(f'after tail: {time.time() - t0:.6f}s')
        return x
    
    def forward_mlp(self, x, img_h, img_w): # x: embedded position coordinates, [n_ray, input_dim]
        '''The input data format is for MLP network. To keep the data preparation the same as before.
        '''
        # torch.cuda.synchronize()
        # t0 = time.time()
        x = x.view(-1, img_h, img_w, x.shape[1]) # [n_img, H, W, input_dim]
        x = x.permute(0, 3, 1, 2) # [n_img, input_dim, H, W]
        # torch.cuda.synchronize()
        # print(f'permute1: {time.time() - t0:.6f}s')
        
        x = self.forward(x) # [n_img, 3, H, W]
        # torch.cuda.synchronize()
        # print(f'forward: {time.time() - t0:.6f}s')
        
        x = x.permute(0, 2, 3, 1) # [n_img, H, W, 3]
        x = x.reshape(-1, 3) # [n_ray, 3]
        
        # torch.cuda.synchronize()
        # print(f'permute2: {time.time() - t0:.6f}s')
        return x

class NeRF_v4(nn.Module):
    '''Spatial sharing. Two rays'''
    def __init__(self, args, near, far):
        super(NeRF_v4, self).__init__()
        self.args = args
        self.near = near
        self.far = far
        assert args.n_sample_per_ray >= 1
        n_sample = args.n_sample_per_ray
        D, W = args.netdepth, args.netwidth

        # positional embedding function
        self.embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        if args.use_viewdirs:
            self.embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

        # network
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D - 1)
        
        input_dim = input_ch + input_ch_views if args.use_viewdirs else input_ch 
        input_dim *= n_sample

        # head
        self.head = nn.Sequential(*[nn.Linear(input_dim, Ws[0]), nn.ReLU(inplace=True)])
        feat_dim = Ws[0]
        
        # body
        body = []
        for i in range(1, D - 1):
            body += [nn.Linear(feat_dim, Ws[i])] # not add relu here, will be added during forward
            feat_dim = Ws[i]
        self.body = nn.Sequential(*body)
        
        # tail
        if args.linear_tail:
            self.tail = nn.Linear(feat_dim, 3)
        else:
            self.tail = nn.Sequential(*[nn.Linear(feat_dim, 3), nn.Sigmoid()])
        
        # branch net to predict neighbor pixels
        branch_input_dim = Ws[args.branch_loc] + input_dim
        branch = [nn.Linear(branch_input_dim, args.branchwidth), nn.ReLU(inplace=True)] # 1st layer
        for _ in range(args.branchdepth - 2):
            branch += [nn.Linear(args.branchwidth, args.branchwidth), nn.ReLU(inplace=True)]
        branch += [nn.Linear(args.branchwidth, 3)] # last layer, predict the residual of rgb, so 3
        self.branch = nn.Sequential(*branch)

    def forward(self, rays_o, rays_d, rays_d2, scale=1, global_step=-1, perturb=0, test=False):
        '''rays_d2: the second ray
        '''
        if test:
            return self._test_forward(rays_o, rays_d, scale, perturb)

        n_ray = rays_o.size(0)
        n_sample = self.args.n_sample_per_ray

        if n_sample > 1:
            t_vals = torch.linspace(0., 1., steps=n_sample).to(device)
            t_vals = t_vals[None, :].expand(n_ray, n_sample)
            z_vals = self.near * (1 - t_vals) + self.far * (t_vals)
            if perturb > 0.:
                # get intervals between samples
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals.shape).to(device) # uniform dist [0, 1)
                z_vals = lower + (upper - lower) * t_rand
        
            # get sample coordinates
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [n_ray, n_sample, 3]
            pts2 = rays_o[..., None, :] + rays_d2[..., None, :] * z_vals[..., :, None] # [n_ray, n_sample, 3]

            # positional embedding
            embedded_pts = self.embed_fn(pts.view(-1, 3)) # shape: [n_ray*n_sample_per_ray, 63]
            embedded_pts = embedded_pts.view(n_ray, -1) # [n_ray, n_sample_per_ray*63]
            embedded_pts2 = self.embed_fn(pts2.view(-1, 3)) # shape: [n_ray*n_sample_per_ray, 63]
            embedded_pts2 = embedded_pts2.view(n_ray, -1) # [n_ray, n_sample_per_ray*63]

            # pose embedding
            if self.args.use_viewdirs:
                raise NotImplementedError
        
        else: # n_sample = 1
            raise NotImplementedError

        # main forward
        head_output = self.head(embedded_pts)
        h = head_output
        for ix, layer in enumerate(self.body):
            h = F.relu(layer(h))
            if ix + 1 == self.args.branch_loc:
                branch_input = h
        h = h + head_output if self.args.use_residual else h
        rgb = self.tail(h)

        # branch forward
        branch_input = torch.cat([branch_input, embedded_pts2], dim=-1)
        branch_input = (rays_d2 - rays_d).norm(dim=-1, keepdim=True) * branch_input * scale
        branch_output = self.branch(branch_input)
        rgb2 = branch_output + rgb
        return rgb, rgb2
    
    def _test_forward(self, rays_o, rays_d, scale, perturb):
        '''rays_o: [n_ray, n_pixel*3], rays_d: [n_ray, n_pixel*3], n_pixel is the num of neibour pixels sharing computation
        '''
        n_ray = rays_o.shape[0]
        rays_d, *rays_d_others = torch.split(rays_d, 3, dim=-1)
        rays_o = rays_o[:, :3] # all the pixels share the same origin, so only using the first 3 is still the same

        n_sample = self.args.n_sample_per_ray
        t_vals = torch.linspace(0., 1., steps=n_sample).to(device)
        t_vals = t_vals[None, :].expand(n_ray, n_sample)
        z_vals = self.near * (1 - t_vals) + self.far * (t_vals)
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device) # uniform dist [0, 1)
            z_vals = lower + (upper - lower) * t_rand
    
        # get sample coordinates
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [n_ray, n_sample, 3]
        
        # positional embedding
        embedded_pts = self.embed_fn(pts.view(-1, 3)) # shape: [n_ray*n_sample_per_ray, 63]
        embedded_pts = embedded_pts.view(n_ray, -1) # [n_ray, n_sample_per_ray*63]

        # main forward
        head_output = self.head(embedded_pts)
        h = head_output
        for ix, layer in enumerate(self.body):
            h = F.relu(layer(h))
            if ix + 1 == self.args.branch_loc:
                branch_input = h
        h = h + head_output if self.args.use_residual else h
        rgb = self.tail(h)

        rgb_others = []
        for rd in rays_d_others: # rays_d_others: [[n_ray, 3], [n_ray, 3], ...]
            pts2 = rays_o[..., None, :] + rd[..., None, :] * z_vals[..., :, None] # [n_ray, n_sample, 3]
            embedded_pts2 = self.embed_fn(pts2.view(-1, 3)) # shape: [n_ray*n_sample_per_ray, 63]
            embedded_pts2 = embedded_pts2.view(n_ray, -1) # [n_ray, n_sample_per_ray*63]

            # branch forward
            branch_input = torch.cat([branch_input, embedded_pts2], dim=-1) # [n_ray, W], [n_ray, 63] -> [n_ray, W+63]
            branch_input = (rd - rays_d).norm(dim=-1, keepdim=True) * branch_input * scale # [n_ray, 1] * [n_ray, W+63] * scalar -> [n_ray, W+63]
            branch_output = self.branch(branch_input) # [n_ray, 3]
            rgb2 = branch_output + rgb # [n_ray, 3]
            rgb_others += [rgb2] # [[n_ray, 3], [n_ray, 3], ...]
        rgbs = [rgb] + rgb_others # [[n_ray, 3], [n_ray, 3], ...]
        rgbs = torch.cat(rgbs, dim=-1) # [n_ray, n_pixel*3]
        return rgbs

NeRF_v5 = NeRF_v3_3 # to maintain back-compatibility
class NeRF_v6(nn.Module):
    '''Based on NeRF_v5, use 3x3 conv'''
    def __init__(self, args, input_dim):
        super(NeRF_v6, self).__init__()
        self.args = args
        D, W = args.netdepth, args.netwidth

        # network width
        if args.layerwise_netwidths:
            Ws = [int(x) for x in args.layerwise_netwidths.split(',')] + [3]
            print('Layer-wise widths are given. Overwrite args.netwidth')
        else:
            Ws = [W] * (D - 1) + [3]
        
        # head
        ks, pd = 1, 0
        self.input_dim = input_dim
        self.head = nn.Sequential(*[nn.Conv2d(in_channels=input_dim, out_channels=Ws[0], kernel_size=ks, padding=pd), nn.ReLU(inplace=True)])
        
        # body
        body = []
        for i in range(1, D - 1):
            body += [nn.Conv2d(in_channels=Ws[i-1], out_channels=Ws[i], kernel_size=ks, padding=pd), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)
        
        # tail
        self.tail = nn.Conv2d(in_channels=Ws[i], out_channels=3, kernel_size=ks, padding=pd) if args.linear_tail \
            else nn.Sequential(*[nn.Conv2d(in_channels=Ws[i], out_channels=3, kernel_size=ks, padding=pd), nn.Sigmoid()])
        
    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x if self.args.use_residual else self.body(x)
        return self.tail(x)