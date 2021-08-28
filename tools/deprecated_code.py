# batchify
for _ in range(args.render_iters):
    rgb, disp = [], []
    for ix in range(0, rays_o.shape[0], chunk):
        with torch.no_grad():
            rgb_, disp_, *_ = model(rays_o[ix: ix+chunk], rays_d[ix: ix+chunk], perturb=perturb)
            rgb += [rgb_]
            disp += [disp_]
    rgb, disp = torch.cat(rgb, dim=0), torch.cat(disp, dim=0)
    rgb_sum += rgb
    disp_sum += disp

# get depth
n_ray, n_sample = H * W, args.n_sample_per_ray
t_vals = torch.linspace(0., 1., steps=n_sample).to(device)
t_vals = t_vals[None, :].expand(n_ray, n_sample)
z_vals = render_kwargs['near'] * (1 - t_vals) + render_kwargs['far'] * (t_vals)
# get positional embedding of all poses
t0 = time.time()
rays_o_all, rays_d_all, embedded_pts_all = [], [], []
for i, c2w in enumerate(render_poses):
    rays_o, rays_d = get_rays1(H, W, focal, c2w[:3, :4]) # rays_o shape: # [H, W, 3]
    rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [n_ray, n_sample, 3]
    embedded_pts = embed_fn(pts.view(-1, 3)) # shape: [n_ray*n_sample_per_ray, 63]
    embedded_pts = embedded_pts.view(rays_o.size(0), -1) # [n_ray, n_sample_per_ray*63]
    embedded_pts_all += [embedded_pts]
print(f'{time.time() - t0:.4f}s -- positional encoding of all poses')


