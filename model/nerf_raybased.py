from os import getgroups
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np, time, math

# TODO: remove this dependency
from torchsearchsorted import searchsorted
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
    '''New idea: one forward to get multi-outputs.
    '''
    def __init__(self, args, near, far, device, print=print):
        super(NeRF_v2, self).__init__()
        self.args = args
        self.near = near
        self.far = far
        D, W = args.netdepth, args.netwidth
        self.skips = [int(x) for x in args.skips.split(',')] if args.skips else []
        self.print = print
        assert args.n_sample_per_ray >= 1

        # positional embedding function
        self.embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        if args.use_viewdirs:
            self.embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

        # head network, get predicted sampled points
        n_sample_per_ray, D_head = args.n_sample_per_ray, args.D_head
        dim_in = input_ch + input_ch_views if args.encode_input else 6 # 6: postion 3 + pose 3
        if D_head == 1:
            head = [nn.Linear(dim_in, n_sample_per_ray), nn.Sigmoid()]
        elif D_head == 2:
            head = [nn.Linear(dim_in, W), nn.ReLU(), nn.Linear(W, n_sample_per_ray), nn.Sigmoid()]
        else:
            head = [nn.Linear(dim_in, W), nn.ReLU()]
            for _ in range(D_head - 2):
                head += [nn.Linear(W, W), nn.ReLU()]
            head += [nn.Linear(W, n_sample_per_ray), nn.Sigmoid()]
        self.head = nn.Sequential(*head)

        # body network
        if args.use_group_conv:
            modules = [nn.Conv2d(input_ch * n_sample_per_ray, W, kernel_size=1, groups=n_sample_per_ray)]
            for i in range(D - 1):
                if i not in self.skips:
                    modules += [nn.Conv2d(W, W, kernel_size=1, groups=n_sample_per_ray)]
                else:
                    modules += [nn.Conv2d(W + input_ch * n_sample_per_ray, W, kernel_size=1, groups=n_sample_per_ray)]
            self.pts_linears = nn.ModuleList(modules)
        else:
            self.pts_linears = nn.ModuleList(
                [nn.Linear(input_ch * n_sample_per_ray, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch * n_sample_per_ray, W) for i in range(D - 1)])
            
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        if self.args.use_group_conv:
            self.views_linears = nn.ModuleList([nn.Conv2d(input_ch_views * n_sample_per_ray + W, W//2, kernel_size=1, groups=n_sample_per_ray)])
        else:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views * n_sample_per_ray + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if args.use_viewdirs:
            if self.args.use_group_conv:
                self.feature_linear = nn.Conv2d(W, W, kernel_size=1, groups=n_sample_per_ray)
                self.alpha_linear = nn.Conv2d(W, n_sample_per_ray, kernel_size=1, groups=n_sample_per_ray)
                self.rgb_linear = nn.Conv2d(W//2, 3 * n_sample_per_ray, kernel_size=1, groups=n_sample_per_ray)
            else:
                self.feature_linear = nn.Linear(W, W)
                self.alpha_linear = nn.Linear(W, n_sample_per_ray)
                self.rgb_linear = nn.Linear(W//2, 3 * n_sample_per_ray)
        else:
            output_ch = 5 if args.N_importance > 0 else 4
            self.output_linear = nn.Linear(W, output_ch)

        # original NeRF forward impl.
        self.network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=self.embed_fn,
                                                                    embeddirs_fn=self.embeddirs_fn,
                                                                    netchunk=args.netchunk)

        # use dropout
        self.dropout_layer = [10000, 10000] # a large int as placeholder
        if self.args.dropout_layer:
            self.dropout_layer = [int(x) for x in self.args.dropout_layer.split(',')]
        
    def forward(self, rays_o, rays_d, global_step=-1, perturb=0, perm_invar=False):
        n_ray = rays_o.size(0)
        n_sample = self.args.n_sample_per_ray

        if n_sample > 1:
            # predict depth of sampled points
            if self.args.learn_pts:
                # set up input
                input = torch.cat([rays_o, rays_d], dim=-1) # [n_ray, 6]
                if self.args.encode_input:
                    embedded_rays_o = self.embed_fn(rays_o)
                    embedded_rays_d = self.embeddirs_fn(rays_d)
                    input = torch.cat([embedded_rays_o, embedded_rays_d], dim=-1) # [n_ray, 90]
                
                # get sample depths
                intervals = self.head(input) / (0.5 * n_sample) # [n_ray, n_sample]
                t_vals = torch.cumsum(intervals, dim=1) # make sure it is in ascending order
                if global_step % self.args.i_print == 0:
                    logtmp = ['%.3f' % x for x in t_vals[0]]
                    self.print('t_vals: ' + ' '.join(logtmp))
                z_vals = self.near * (1 - t_vals) + self.far * t_vals # depth, [n_ray, n_sample]
            else:
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
            if perm_invar: # permutation invariance prior
                rand_index = torch.randperm(n_sample)
                pts = pts[:, rand_index]

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
                dirs = viewdirs[:, None].expand(pts.shape) # [n_ray, 3] -> [n_ray, n_sample_per_ray, 3]

                if perm_invar: # permutation invariance prior
                    dirs = dirs[:, rand_index]

                dirs = torch.reshape(dirs, (-1, 3))
                embedded_dirs = self.embeddirs_fn(dirs)
                embedded_dirs = embedded_dirs.view(rays_o.size(0), -1) # [n_ray, n_sample_per_ray * 27]
        
        else: # n_sample = 1
            embedded_pts = self.embed_fn(rays_o) # [n_ray, 63]
            embedded_dirs = self.embeddirs_fn(rays_d / rays_d.norm(dim=-1, keepdim=True)) # [n_ray, 27]

        # body network
        if self.args.use_group_conv:
            h = embedded_pts
            h = h[..., None, None]
            for i, layer in enumerate(self.pts_linears):
                h = F.relu(layer(h))
                if i in self.skips:
                    h = h.view(n_ray, n_sample, -1)
                    embedded_pts = embedded_pts.view(n_ray, n_sample, -1)
                    h = torch.cat([h, embedded_pts], dim=-1) # [n_ray, n_sample_per_ray, -1]
                    h = h.view(n_ray, -1, 1, 1) # [n_ray, W + n_sample_per_ray * input_ch, 1, 1]
            # get raw outputs
            if self.args.use_viewdirs:
                alpha = self.alpha_linear(h)[..., 0, 0] # [n_ray, n_sample_per_ray]
                feature = self.feature_linear(h) # [n_ray, W + n_sample_per_ray * input_ch_views, 1, 1]

                feature = feature.view(n_ray, n_sample, -1)
                embedded_dirs = embedded_dirs.view(n_ray, n_sample, -1)
                h = torch.cat([feature, embedded_dirs], dim=-1)
                h = h.view(n_ray, -1, 1, 1)
            
                for i, l in enumerate(self.views_linears):
                    h = self.views_linears[i](h)
                    h = F.relu(h)

                rgb = self.rgb_linear(h)[..., 0, 0]
                raw = torch.cat([rgb, alpha], dim=-1)
                raw = raw.view(n_ray, -1, 4) # [n_ray, n_sample_per_ray, 4]
        else:
            h = embedded_pts
            for i, layer in enumerate(self.pts_linears):
                h = F.relu(layer(h))
                if i >= self.dropout_layer[0]:
                    h = F.dropout(h, p=self.args.dropout_ratio)
                if i in self.skips:
                    h = torch.cat([embedded_pts, h], dim=-1)

            # get raw outputs
            if self.args.use_viewdirs:
                alpha = self.alpha_linear(h) # [n_ray, n_sample_per_ray]
                feature = self.feature_linear(h)
                h = torch.cat([feature, embedded_dirs], -1)
                for i, layer in enumerate(self.views_linears):
                    h = F.relu(layer(h))
                    if i >= self.dropout_layer[1]:
                        h = F.dropout(h, p=self.args.dropout_ratio)
                rgb = self.rgb_linear(h)
                raw = torch.cat([rgb, alpha], dim=-1)
                raw = raw.view(n_ray, -1, 4) # [n_ray, n_sample_per_ray, 4]
        
        if perm_invar:
            inv_rand_index = torch.argsort(rand_index)
            raw = raw[:, inv_rand_index]

        # rendering
        if self.args.directly_predict_rgb:
            rgb_map = torch.sigmoid(raw[..., :3].mean(dim=1)) # [n_ray, 3]
            disp_map = rgb_map # placeholder
            return rgb_map, disp_map
        else: # use rendering equation
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, self.args.raw_noise_std, white_bkgd=False, pytest=False, global_step=global_step, print=self.print)
            return rgb_map, disp_map, acc_map, weights, depth_map, raw, pts, viewdirs
    
    def perm_invar_forward(self, rays_o, rays_d, global_step=-1, perturb=0):
        rgb_map, disp_map, acc_map, weights, depth_map, raw, pts, viewdirs = self.forward(rays_o, rays_d, global_step=global_step, perturb=perturb)
        raws = []
        for _ in range(self.args.n_perm_invar):
            out = self.forward(rays_o, rays_d, global_step=-1, perturb=perturb, perm_invar=True)
            raws.append(out[5])
        raws = torch.cat(raws, dim=0) # [n_perm_invar, n_ray, n_sample, 4]
        loss_perm_invar = torch.var(raws, dim=0).mean()

        # print log
        # if global_step % self.args.i_print == 0:
        #     raws = raws.view(self.args.n_perm_invar, -1)
        #     for i in range(self.args.n_perm_invar):
        #         logtmp = ['%.3f' % x for x in raws[i]]
        #         self.print(' '.join(logtmp))
        return rgb_map, disp_map, acc_map, weights, depth_map, raw, pts, viewdirs, loss_perm_invar

class NeRF_v3(nn.Module):
    '''New idea: Use conv layers.
    '''
    def __init__(self, args, near, far, device, print=print):
        super(NeRF_v3, self).__init__()
        self.args = args
        self.near = near
        self.far = far
        D, W = args.netdepth, args.netwidth
        self.skips = [int(x) for x in args.skips.split(',')] if args.skips else []
        self.print = print
        assert args.n_sample_per_ray >= 1

        # positional embedding function
        self.embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        if args.use_viewdirs:
            self.embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

        # head network, get predicted sampled points
        n_sample_per_ray, D_head = args.n_sample_per_ray, args.D_head
        dim_in = input_ch + input_ch_views if args.encode_input else 6 # 6: postion 3 + pose 3
        if D_head == 1:
            head = [nn.Linear(dim_in, n_sample_per_ray), nn.Sigmoid()]
        elif D_head == 2:
            head = [nn.Linear(dim_in, W), nn.ReLU(), nn.Linear(W, n_sample_per_ray), nn.Sigmoid()]
        else:
            head = [nn.Linear(dim_in, W), nn.ReLU()]
            for _ in range(D_head - 2):
                head += [nn.Linear(W, W), nn.ReLU()]
            head += [nn.Linear(W, n_sample_per_ray), nn.Sigmoid()]
        self.head = nn.Sequential(*head)

        # body network
        if args.use_group_conv:
            modules = [nn.Conv2d(input_ch * n_sample_per_ray, W, kernel_size=1, groups=n_sample_per_ray)]
            for i in range(D - 1):
                if i not in self.skips:
                    modules += [nn.Conv2d(W, W, kernel_size=1, groups=n_sample_per_ray)]
                else:
                    modules += [nn.Conv2d(W + input_ch * n_sample_per_ray, W, kernel_size=1, groups=n_sample_per_ray)]
            self.pts_linears = nn.ModuleList(modules)
        else:
            self.pts_linears = nn.ModuleList(
                [nn.Linear(input_ch * n_sample_per_ray, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch * n_sample_per_ray, W) for i in range(D - 1)])
            
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        if self.args.use_group_conv:
            self.views_linears = nn.ModuleList([nn.Conv2d(input_ch_views * n_sample_per_ray + W, W//2, kernel_size=1, groups=n_sample_per_ray)])
        else:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views * n_sample_per_ray + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if args.use_viewdirs:
            if self.args.use_group_conv:
                self.feature_linear = nn.Conv2d(W, W, kernel_size=1, groups=n_sample_per_ray)
                self.alpha_linear = nn.Conv2d(W, n_sample_per_ray, kernel_size=1, groups=n_sample_per_ray)
                self.rgb_linear = nn.Conv2d(W//2, 3 * n_sample_per_ray, kernel_size=1, groups=n_sample_per_ray)
            else:
                self.feature_linear = nn.Linear(W, W)
                self.alpha_linear = nn.Linear(W, n_sample_per_ray)
                self.rgb_linear = nn.Linear(W//2, 3 * n_sample_per_ray)
        else:
            output_ch = 5 if args.N_importance > 0 else 4
            self.output_linear = nn.Linear(W, output_ch)

        # original NeRF forward impl.
        self.network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                    embed_fn=self.embed_fn,
                                                                    embeddirs_fn=self.embeddirs_fn,
                                                                    netchunk=args.netchunk)

        # use dropout
        self.dropout_layer = [10000, 10000] # a large int as placeholder
        if self.args.dropout_layer:
            self.dropout_layer = [int(x) for x in self.args.dropout_layer.split(',')]
        
    def forward(self, rays_o, rays_d, global_step=-1, perturb=0, perm_invar=False):
        n_ray = rays_o.size(0)
        n_sample = self.args.n_sample_per_ray

        if n_sample > 1:
            # predict depth of sampled points
            if self.args.learn_pts:
                # set up input
                input = torch.cat([rays_o, rays_d], dim=-1) # [n_ray, 6]
                if self.args.encode_input:
                    embedded_rays_o = self.embed_fn(rays_o)
                    embedded_rays_d = self.embeddirs_fn(rays_d)
                    input = torch.cat([embedded_rays_o, embedded_rays_d], dim=-1) # [n_ray, 90]
                
                # get sample depths
                intervals = self.head(input) / (0.5 * n_sample) # [n_ray, n_sample]
                t_vals = torch.cumsum(intervals, dim=1) # make sure it is in ascending order
                if global_step % self.args.i_print == 0:
                    logtmp = ['%.3f' % x for x in t_vals[0]]
                    self.print('t_vals: ' + ' '.join(logtmp))
                z_vals = self.near * (1 - t_vals) + self.far * t_vals # depth, [n_ray, n_sample]
            else:
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
            if perm_invar: # permutation invariance prior
                rand_index = torch.randperm(n_sample)
                pts = pts[:, rand_index]

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
                dirs = viewdirs[:, None].expand(pts.shape) # [n_ray, 3] -> [n_ray, n_sample_per_ray, 3]

                if perm_invar: # permutation invariance prior
                    dirs = dirs[:, rand_index]

                dirs = torch.reshape(dirs, (-1, 3))
                embedded_dirs = self.embeddirs_fn(dirs)
                embedded_dirs = embedded_dirs.view(rays_o.size(0), -1) # [n_ray, n_sample_per_ray * 27]
        
        else: # n_sample = 1
            embedded_pts = self.embed_fn(rays_o) # [n_ray, 63]
            embedded_dirs = self.embeddirs_fn(rays_d / rays_d.norm(dim=-1, keepdim=True)) # [n_ray, 27]

        # body network
        if self.args.use_group_conv:
            h = embedded_pts
            h = h[..., None, None]
            for i, layer in enumerate(self.pts_linears):
                h = F.relu(layer(h))
                if i in self.skips:
                    h = h.view(n_ray, n_sample, -1)
                    embedded_pts = embedded_pts.view(n_ray, n_sample, -1)
                    h = torch.cat([h, embedded_pts], dim=-1) # [n_ray, n_sample_per_ray, -1]
                    h = h.view(n_ray, -1, 1, 1) # [n_ray, W + n_sample_per_ray * input_ch, 1, 1]
            # get raw outputs
            if self.args.use_viewdirs:
                alpha = self.alpha_linear(h)[..., 0, 0] # [n_ray, n_sample_per_ray]
                feature = self.feature_linear(h) # [n_ray, W + n_sample_per_ray * input_ch_views, 1, 1]

                feature = feature.view(n_ray, n_sample, -1)
                embedded_dirs = embedded_dirs.view(n_ray, n_sample, -1)
                h = torch.cat([feature, embedded_dirs], dim=-1)
                h = h.view(n_ray, -1, 1, 1)
            
                for i, l in enumerate(self.views_linears):
                    h = self.views_linears[i](h)
                    h = F.relu(h)

                rgb = self.rgb_linear(h)[..., 0, 0]
                raw = torch.cat([rgb, alpha], dim=-1)
                raw = raw.view(n_ray, -1, 4) # [n_ray, n_sample_per_ray, 4]
        else:
            h = embedded_pts
            for i, layer in enumerate(self.pts_linears):
                h = F.relu(layer(h))
                if i >= self.dropout_layer[0]:
                    h = F.dropout(h, p=self.args.dropout_ratio)
                if i in self.skips:
                    h = torch.cat([embedded_pts, h], dim=-1)

            # get raw outputs
            if self.args.use_viewdirs:
                alpha = self.alpha_linear(h) # [n_ray, n_sample_per_ray]
                feature = self.feature_linear(h)
                h = torch.cat([feature, embedded_dirs], -1)
                for i, layer in enumerate(self.views_linears):
                    h = F.relu(layer(h))
                    if i >= self.dropout_layer[1]:
                        h = F.dropout(h, p=self.args.dropout_ratio)
                rgb = self.rgb_linear(h)
                raw = torch.cat([rgb, alpha], dim=-1)
                raw = raw.view(n_ray, -1, 4) # [n_ray, n_sample_per_ray, 4]
        
        if perm_invar:
            inv_rand_index = torch.argsort(rand_index)
            raw = raw[:, inv_rand_index]

        # rendering
        if self.args.directly_predict_rgb:
            rgb_map = torch.sigmoid(raw[..., :3].mean(dim=1)) # [n_ray, 3]
            disp_map = rgb_map # placeholder
            return rgb_map, disp_map
        else: # use rendering equation
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, self.args.raw_noise_std, white_bkgd=False, pytest=False, global_step=global_step, print=self.print)
            return rgb_map, disp_map, acc_map, weights, depth_map, raw, pts, viewdirs
    
    def perm_invar_forward(self, rays_o, rays_d, global_step=-1, perturb=0):
        rgb_map, disp_map, acc_map, weights, depth_map, raw, pts, viewdirs = self.forward(rays_o, rays_d, global_step=global_step, perturb=perturb)
        raws = []
        for _ in range(self.args.n_perm_invar):
            out = self.forward(rays_o, rays_d, global_step=-1, perturb=perturb, perm_invar=True)
            raws.append(out[5])
        raws = torch.cat(raws, dim=0) # [n_perm_invar, n_ray, n_sample, 4]
        loss_perm_invar = torch.var(raws, dim=0).mean()

        # print log
        # if global_step % self.args.i_print == 0:
        #     raws = raws.view(self.args.n_perm_invar, -1)
        #     for i in range(self.args.n_perm_invar):
        #         logtmp = ['%.3f' % x for x in raws[i]]
        #         self.print(' '.join(logtmp))
        return rgb_map, disp_map, acc_map, weights, depth_map, raw, pts, viewdirs, loss_perm_invar