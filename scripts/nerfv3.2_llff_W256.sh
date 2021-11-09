# room
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/room_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__room__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200

# fern
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/fern_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/fern_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__fern__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200

# leaves
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/leaves_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/leaves_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__leaves__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200

# fortress
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/fortress_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/fortress_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__fortress__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200

# orchids
CUDA_VISIBLE_DEVICES=4 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/orchids_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/orchids_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__orchids__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200

# flower
CUDA_VISIBLE_DEVICES=5 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/flower_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/flower_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__flower__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200

# trex
CUDA_VISIBLE_DEVICES=6 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/trex_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/trex_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__trex__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200

# horns
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/horns_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/horns_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 120 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__horns__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200