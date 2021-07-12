import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, n_pose=40, perturb=False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            # print(frame.keys()) # frame keys: file_path, rotation, transform_matrix
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # @mst: add noise jittering
    if isinstance(n_pose, int):
        thetas = np.linspace(-180, 180, n_pose + 1)
        phis = [-30]
    else:
        thetas = np.linspace(-180, 180, n_pose[0] + 1)
        phis = np.linspace(-90, 0, n_pose[1] + 1)
    
    if perturb:
        lower, upper = thetas[..., :-1], thetas[..., 1:]
        rand = torch.rand(lower.shape).data.cpu().numpy() # uniform dist [0, 1)
        thetas = lower + (upper - lower) * rand
    else:
        thetas = thetas[:-1]

    if not isinstance(n_pose, int):
        if perturb:
            lower, upper = phis[..., :-1], phis[..., 1:]
            rand = torch.rand(lower.shape).data.cpu().numpy() # uniform dist [0, 1)
            phis = lower + (upper - lower) * rand
        else:
            phis = phis[:-1]

    render_poses = torch.stack([pose_spherical(t, p, 4) for p in phis for t in thetas], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split

def setup_blender_datadir(datadir_old, datadir_new):
    import shutil
    if os.path.exists(datadir_new):
        if os.path.isfile(datadir_new): 
            os.remove(datadir_new)
        else:
            shutil.rmtree(datadir_new)
        os.makedirs(datadir_new)
    
    # copy json file
    shutil.copy(datadir_old + '/transforms_train.json', datadir_new)
    
    # create softlink for images
    imgs = [x for x in os.listdir(datadir_old + '/train') if x.endswith('.png')]
    os.makedirs(datadir_new + '/train')
    cwd = os.getcwd()
    os.chdir(datadir_new + '/train')
    dirname_old = datadir_old.split('/')[-1]
    for img in imgs:
        # print(os.getcwd())
        # print(f'../../{dirname_old}/train/{img} -> f{img}')
        os.symlink(f'../../{dirname_old}/train/{img}', f'{img}')
    os.chdir(cwd) # change back working directory
    
def save_blender_data(datadir, poses, images, split='train'):
    '''Save pseudo data created by a trained nerf.'''
    import json, imageio, os
    json_file = '%s/transforms_%s.json' % (datadir, split)
    with open(json_file) as f:
        data = json.load(f)
    
    frames = data['frames']
    n_img = len(frames)
    folder = os.path.split(json_file)[0] # example: 'data/nerf_synthetic/lego/'
    for pose, img in zip(poses, images):
        n_img += 1
        img_path = './%s/r_%d_pseudo' % (split, n_img-1) # add the 'pseudo' suffix to differentiate it from real-world data
        
        # add new frame data to json
        new_frame = {k:v for k,v in frames[0].items()}
        new_frame['file_path'] = img_path
        new_frame['transform_matrix'] = pose.data.cpu().numpy().tolist()
        frames += [new_frame]

        # save image
        img_path = '%s/%s.png' % (folder, img_path)
        imageio.imwrite(img_path, to8b(img.data.cpu().numpy()))

    with open(json_file, 'w') as f:
        data['frames'] = frames
        json.dump(data, f, indent=4)

def load_blender_data_v2(datadir, half_res=False, white_bkgd=True, split='train'):
    '''Load data with psuedo data'''
    with open(os.path.join(datadir, 'transforms_{}.json'.format(split)), 'r') as fp:
        meta = json.load(fp)

    poses, imgs = [], []
    for ix, frame in enumerate(meta['frames']):
        fname = os.path.join(datadir, frame['file_path'] + '.png')
        img = np.array(imageio.imread(fname)).astype(np.float32) / 255.
        pose = np.array(frame['transform_matrix']).astype(np.float32)
        
        if 'pseudo' not in fname: # real-world data
            if half_res:
                H, W = img.shape[:2]
                img = cv2.resize(img, (H//2, W//2), interpolation=cv2.INTER_AREA)
            img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:]) if white_bkgd else img[..., :3] 
                
        imgs.append(img)
        poses.append(pose)
    return np.array(imgs), np.array(poses)