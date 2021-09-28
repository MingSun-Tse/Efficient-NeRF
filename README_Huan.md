
```
CUDA_VISIBLE_DEVICES=0 python create_data.py --create_data 3x3rays --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v11_3x3rays_queue_NoRandFocal_10000Images --screen --project nerf__lego__CreateData_v11_3x3rays_queue_NoRandFocal_10000Images --cache_ignore data,Experiments,Experiments2,torchsearchsorted --no_rand_focal --max_save 10000000000
```