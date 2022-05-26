CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/lego_noview.txt --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --cache_ignore data,__pycache__,torchsearchsorted,imgs --screen --trial.ON --trial.body_arch resmlp --num_worker 12 --warmup_lr 0.0001,200 --plucker --project nerfv3.2__lego__W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__Plucker