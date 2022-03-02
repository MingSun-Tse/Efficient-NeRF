import os, time, numpy as np, sys, configargparse
import torch
import imageio 
import json
import cv2
to_tensor = lambda x: x.to('cpu') if isinstance(x, torch.Tensor) else torch.Tensor(x).to('cpu')
to_array = lambda x: x if isinstance(x, np.ndarray) else x.data.cpu().numpy()

def get_rays(H, W, focal, c2w, trans_origin='', focal_scale=1):
    focal *= focal_scale
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)) # pytorch's meshgrid has indexing='ij'
    i, j = i.t(), j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1) # TODO-@mst: check if this H/W or W/H is in the right order
    # Rotate ray directions from camera frame to the world frame
    # rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(dim=-2) * c2w[:3,:3], -1)  # shape: [H, W, 3]
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

r"""Usage:
        python <this_file> --splits <train_val_splits> --datadir <dir_path_to_original_data>
Example: 
        python convert_original_data_to_rays.py --splits train --datadir data/nerf_synthetic/lego
"""

############################################## Input Args
half_res = True # default setting, corresponding to 400x400 images in the synthetic dataset in NeRF
white_bkgd = True # default setting for the synthetic dataset in NeRF
split_size = 4096 # manually set
##############################################

parser = configargparse.ArgumentParser()
parser.add_argument("--splits", type=str, default='')
parser.add_argument("--datadir", type=str, default='')
parser.add_argument("--suffix", type=str, default='')
parser.add_argument("--ignore", type=str, default='', help='ignore some samples')
args = parser.parse_args()

# Hand-designed rule
if 'ficus' in args.datadir:
    args.ignore = '10,14,24,26,30,31,37,39,47,48,54,55,57,58,66,67,74,75,76,77,81,82,87,88,89,94,97,99'

# Set up save folders
splits = args.splits.split(',')
datadir = args.datadir
prefix = ''.join(splits)
suffix = sys.argv[3] if len(sys.argv) == 4 else ''
savedir = f'{os.path.normpath(datadir)}_{prefix}_Rand_Origins_Dirs_{split_size}RaysPerNpy{args.suffix}'
os.makedirs(savedir, exist_ok=True)

# Load all train/val images
all_imgs, all_poses, metas = [], [], {}
ignored_img_indices = args.ignore.split(',')
for s in splits:
    with open(os.path.join(datadir, 'transforms_{}.json'.format(s)), 'r') as fp:
        metas[s] = json.load(fp)
for s in splits:
    meta = metas[s]
    imgs = []
    poses = []
    for frame in meta['frames']:
        fname = os.path.join(datadir, frame['file_path'] + '.png')
        img_index = frame['file_path'].split('_')[-1] # 'file_path' example: "./train/r_3"
        if img_index not in ignored_img_indices:
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    num_channels = imgs[-1].shape[2] # @mst: for donerf data, some of them do not have A channel
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_poses.append(poses)
all_imgs = np.concatenate(all_imgs, 0)
all_poses = np.concatenate(all_poses, 0)
print(f'Read all images and poses, done. all_imgs shape {all_imgs.shape}, all_poses shape {all_poses.shape}')

# Resize if necessary
H, W = all_imgs[0].shape[:2]
camera_angle_x = float(meta['camera_angle_x'])
focal = .5 * W / np.tan(.5 * camera_angle_x)
if half_res:
    H = H // 2
    W = W // 2
    focal = focal / 2.
    imgs_half_res = np.zeros((all_imgs.shape[0], H, W, num_channels))
    for i, img in enumerate(all_imgs):
        imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
    all_imgs = imgs_half_res
all_imgs = to_tensor(all_imgs)
all_poses = to_tensor(all_poses)
if white_bkgd:
    all_imgs = all_imgs[..., :3] * all_imgs[..., -1:] + (1. - all_imgs[..., -1:])
print(f'Resize, done. all_imgs shape {all_imgs.shape}, all_poses shape {all_poses.shape}')

# Get rays together
all_data = [] # All rays_o, rays_d, rgb will be saved here
for im, po in zip(all_imgs, all_poses):
    rays_o, rays_d = get_rays(H, W, focal, po[:3, :4]) # [H, W, 3]
    data = torch.cat([rays_o, rays_d, im], dim=-1) # [H, W, 9]
    all_data += [data.view(H*W, 9)] # [H*W, 9]
all_data = torch.cat(all_data, dim=0)
print(f'Collect all rays, done. all_data shape {all_data.shape}')

# Shuffle rays
rand_ix1 = np.random.permutation(all_data.shape[0])
rand_ix2 = np.random.permutation(all_data.shape[0])
all_data = all_data[rand_ix1][rand_ix2]
all_data = to_array(all_data)

# Save
split = 0
num = all_data.shape[0] // split_size * split_size
for ix in range(0, num, split_size):
    split += 1
    save_path = f'{savedir}/{prefix}_{split}.npy'
    d = all_data[ix: ix+split_size]
    np.save(save_path, d)
    print(f'[{split}/{num//split_size}] save_path: {save_path}')
print(f'All data saved at "{savedir}"')