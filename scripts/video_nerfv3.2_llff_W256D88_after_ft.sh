# room
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/room_noview.txt --pretrained_ckpt Experiments/nerfv3.2__llff_room__S16W256D88*SERVER-20220305*/weights/ckpt_best.tar --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual  --trial.ON --trial.body_arch resmlp --project Video__nerfv3.2__llff_room__AfterFT --cache_ignore data --render_only  --screen

# fern
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/fern_noview.txt --pretrained_ckpt Experiments/nerfv3.2__llff_fern__S16W256D88*SERVER-20220305*/weights/ckpt_best.tar --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual  --trial.ON --trial.body_arch resmlp --project Video__nerfv3.2__llff_fern__AfterFT --cache_ignore data --render_only  --screen

# leaves
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/leaves_noview.txt --pretrained_ckpt Experiments/nerfv3.2__llff_leaves__S16W256D88*SERVER-20220305*/weights/ckpt_best.tar --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual  --trial.ON --trial.body_arch resmlp --project Video__nerfv3.2__llff_leaves__AfterFT --cache_ignore data --render_only  --screen

# fortress
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/fortress_noview.txt --pretrained_ckpt Experiments/nerfv3.2__llff_fortress__S16W256D88*SERVER-20220305*/weights/ckpt_best.tar --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual  --trial.ON --trial.body_arch resmlp --project Video__nerfv3.2__llff_fortress__AfterFT --cache_ignore data --render_only  --screen

# orchids
CUDA_VISIBLE_DEVICES=4 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/orchids_noview.txt --pretrained_ckpt Experiments/nerfv3.2__llff_orchids__S16W256D88*SERVER-20220305*/weights/ckpt_best.tar --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual  --trial.ON --trial.body_arch resmlp --project Video__nerfv3.2__llff_orchids__AfterFT --cache_ignore data --render_only  --screen

# flower
CUDA_VISIBLE_DEVICES=5 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/flower_noview.txt --pretrained_ckpt Experiments/nerfv3.2__llff_flower__S16W256D88*SERVER-20220305*/weights/ckpt_best.tar --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual  --trial.ON --trial.body_arch resmlp --project Video__nerfv3.2__llff_flower__AfterFT --cache_ignore data --render_only  --screen

# trex
CUDA_VISIBLE_DEVICES=6 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/trex_noview.txt --pretrained_ckpt Experiments/nerfv3.2__llff_trex__S16W256D88*SERVER-20220305*/weights/ckpt_best.tar --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual  --trial.ON --trial.body_arch resmlp --project Video__nerfv3.2__llff_trex__AfterFT --cache_ignore data --render_only  --screen

# horns
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/horns_noview.txt --pretrained_ckpt Experiments/nerfv3.2__llff_horns__S16W256D88*SERVER-20220305*/weights/ckpt_best.tar --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual  --trial.ON --trial.body_arch resmlp --project Video__nerfv3.2__llff_horns__AfterFT --cache_ignore data --render_only  --screen