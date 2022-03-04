# sanmiguel
tail -n 20 data/sanmiguel/dataset_info.json
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/donerf_sanmiguel_noview.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 88 --datadir_kd data/donerf_data/sanmiguel_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__donerf_sanmiguel__S16W181D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200 --given_render_path_rays data/donerf_data/sanmiguel/test_rays.pt --testskip 1 --i_video 1000000000000 --trial.near 1.3081911981105805 --trial.far 27.812076473236086


# pavillon
tail -n 20 data/pavillon/dataset_info.json
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/donerf_pavillon_noview.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 88 --datadir_kd data/donerf_data/pavillon_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__donerf_pavillon__S16W181D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Far8 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200 --given_render_path_rays data/donerf_data/pavillon/test_rays.pt --testskip 1 --i_video 1000000000000 --trial.near 1.1224385499954224 --trial.far 8 # 118.75775413513185


# classroom
tail -n 20 data/classroom/dataset_info.json
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/donerf_classroom_noview.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 88 --datadir_kd data/donerf_data/classroom_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__donerf_classroom__S16W181D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200 --given_render_path_rays data/donerf_data/classroom/test_rays.pt --testskip 1 --i_video 1000000000000 --trial.near 0.6359080076217651 --trial.far 8.79825210571289


# bulldozer
tail -n 20 data/bulldozer/dataset_info.json
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/donerf_bulldozer_noview.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 88 --datadir_kd data/donerf_data/bulldozer_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__donerf_bulldozer__S16W181D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200 --given_render_path_rays data/donerf_data/bulldozer/test_rays.pt --testskip 1 --i_video 1000000000000 --trial.near 0.5999020725488663 --trial.far 3.6212123036384583 


# forest
tail -n 20 data/forest/dataset_info.json
CUDA_VISIBLE_DEVICES=4 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/donerf_forest_noview.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 88 --datadir_kd data/donerf_data/forest_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__donerf_forest__S16W181D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Near20Far40 --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200 --given_render_path_rays data/donerf_data/forest/test_rays.pt --testskip 1 --i_video 1000000000000 --trial.near 20 --trial.far 40 # --trial.near 5.79003734588623 --trial.far 1293.1921508789062


# barbershop
tail -n 20 data/barbershop/dataset_info.json
CUDA_VISIBLE_DEVICES=5 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/donerf_barbershop_noview.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 88 --datadir_kd data/donerf_data/barbershop_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__donerf_barbershop__S16W181D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --i_update_data 10000 --warmup_lr 0.0001,200 --given_render_path_rays data/donerf_data/barbershop/test_rays.pt --testskip 1 --i_video 1000000000000 --trial.near 0.31855402439832686 --trial.far 8.704841423034669