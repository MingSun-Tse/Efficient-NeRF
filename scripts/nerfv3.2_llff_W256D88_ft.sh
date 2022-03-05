# room
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/room_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/room_train_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 120 --N_iters 1600000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__llff_room__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Resume_OriginalTrain --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200     --pretrained Experiments/nerfv3.2__room*SERVER-2021111*/weights/ckpt.tar --resume --i_test 200 --i_weights 200 --i_print 50 --test_pretrained

# fern
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/fern_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/fern_train_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 120 --N_iters 1600000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__llff_fern__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Resume_OriginalTrain --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200     --pretrained Experiments/nerfv3.2__fern*SERVER-2021111*/weights/ckpt.tar --resume --i_test 200 --i_weights 200 --i_print 50 --test_pretrained

# leaves
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/leaves_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/leaves_train_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 120 --N_iters 1600000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__llff_leaves__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Resume_OriginalTrain --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200     --pretrained Experiments/nerfv3.2__leaves*SERVER-2021111*/weights/ckpt.tar --resume --i_test 200 --i_weights 200 --i_print 50 --test_pretrained

# fortress
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/fortress_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/fortress_train_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 120 --N_iters 1600000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__llff_fortress__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Resume_OriginalTrain --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200     --pretrained Experiments/nerfv3.2__fortress*SERVER-2021111*/weights/ckpt.tar --resume --i_test 200 --i_weights 200 --i_print 50 --test_pretrained

# orchids
CUDA_VISIBLE_DEVICES=4 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/orchids_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/orchids_train_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 120 --N_iters 1600000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__llff_orchids__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Resume_OriginalTrain --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200     --pretrained Experiments/nerfv3.2__orchids*SERVER-2021111*/weights/ckpt.tar --resume --i_test 200 --i_weights 200 --i_print 50 --test_pretrained

# flower
CUDA_VISIBLE_DEVICES=5 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/flower_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/flower_train_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 120 --N_iters 1600000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__llff_flower__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Resume_OriginalTrain --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200     --pretrained Experiments/nerfv3.2__flower*SERVER-2021111*/weights/ckpt.tar --resume --i_test 200 --i_weights 200 --i_print 50 --test_pretrained

# trex
CUDA_VISIBLE_DEVICES=6 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/trex_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/trex_train_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 120 --N_iters 1600000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__llff_trex__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Resume_OriginalTrain --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200     --pretrained Experiments/nerfv3.2__trex*SERVER-2021111*/weights/ckpt.tar --resume --i_test 200 --i_weights 200 --i_print 50 --test_pretrained

# horns
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/horns_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_llff_data/horns_train_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 120 --N_iters 1600000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__llff_horns__S16W256D88_DPRGB_Res_NoDir_ResMLPBody__DataV8_BS98304_Hard0.2_20xBS__WarmupLR__Resume_OriginalTrain --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200     --pretrained Experiments/nerfv3.2__horns*SERVER-2021111*/weights/ckpt.tar --resume --i_test 200 --i_weights 200 --i_print 50 --test_pretrained