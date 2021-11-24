# chair
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/chair_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/chair_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project Video__nerfv3.2__chair__NPose40 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200     --pretrained_ckpt Experiments/nerfv3.2__chair__S16W256D88*SERVER-202111*/weights/ckpt.tar --render_only 

# drums
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/drums_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/drums_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project Video__nerfv3.2__drums__NPose40 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200     --pretrained_ckpt Experiments/nerfv3.2__drums__S16W256D88*SERVER-202111*/weights/ckpt.tar --render_only 

# ficus
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/ficus_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/ficus_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project Video__nerfv3.2__ficus__NPose40 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200     --pretrained_ckpt Experiments/nerfv3.2__ficus__S16W256D88*SERVER-202111*/weights/ckpt.tar --render_only 

# hotdog
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/hotdog_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/hotdog_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project Video__nerfv3.2__hotdog__NPose40 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200     --pretrained_ckpt Experiments/nerfv3.2__hotdog__S16W256D88*SERVER-202111*/weights/ckpt.tar --render_only 

# lego
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project Video__nerfv3.2__lego__NPose40 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200     --pretrained_ckpt Experiments/nerfv3.2__lego__S16W256D88*SERVER-202111*/weights/ckpt.tar --render_only 

# materials
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/materials_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/materials_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project Video__nerfv3.2__materials__NPose40 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200     --pretrained_ckpt Experiments/nerfv3.2__materials__S16W256D88*SERVER-202111*/weights/ckpt.tar --render_only 

# mic
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/mic_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/mic_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project Video__nerfv3.2__mic__NPose40 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200     --pretrained_ckpt Experiments/nerfv3.2__mic__S16W256D88*SERVER-202111*/weights/ckpt.tar --render_only 

# ship
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/ship_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/ship_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project Video__nerfv3.2__ship__NPose40 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200     --pretrained_ckpt Experiments/nerfv3.2__ship__S16W256D88*SERVER-202111*/weights/ckpt.tar --render_only 