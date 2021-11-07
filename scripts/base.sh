## Train: nerf, lego
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/lego.txt --project nerf__lego --i_print 10 --i_weights 10 --i_video 10 --debug
# make nerf_v2 back-compatible
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --config configs/lego.txt --model_name nerf --project nerf__lego --i_print 10 --i_weights 10 --i_video 10 --debug
# better nerf
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --config configs/lego.txt --model_name nerf --N_importance 256 --N_samples 128 --project nerf__lego__NSample128+256 --screen
# run nerf on DONERF data
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --config configs/lego.txt --model_name nerf --datadir data/nerf_synthetic/bulldozer --screen --n_pose_video 20,1,1 --testskip 1 --cache_ignore data --project nerf__DONERF_bulldozer

## Train: nerf, ship
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/ship.txt --project nerf__ship --screen --debug

## Train: nerf_v2, lego
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 100 --kd_with_render_pose_mode all_render_pose --kd_poses_update 10000 --n_pose_video 100 --video_poses_perturb --N_iters 400000 --N_rand 16384 --precrop_iters -1 --pretrained_ckpt Experiments/nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_KDWRenderPose100All_BS16384_SERVER142-20210704-150540/*/200000.tar --resume --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_KDWRenderPose100All_BS16384__Resume150540Iter400000

CUDA_VISIBLE_DEVICES=1 nohup python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 100,25 --kd_with_render_pose_mode all_render_pose --n_pose_video 40,3 --video_poses_perturb --N_iters 600000 --N_rand 16384 --precrop_iters -1 --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_KDWRenderPose100,25All_BS16384 > /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 20,5 --kd_with_render_pose_mode all_render_pose --n_pose_video 40,3 --video_poses_perturb --N_iters 600000 --N_rand 16384 --precrop_iters -1 --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_KDWRenderPose20,5All_BS16384 > /dev/null &

# create data (.npy) lego
CUDA_VISIBLE_DEVICES=2 python run_nerf_create_data.py --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 100,25,1 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v5_NPose100,25,1 --screen --project nerf__lego__CreateData_NPose100,25,1

CUDA_VISIBLE_DEVICES=3 python run_nerf_create_data.py --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 100,25,1 --focal_scale 2 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v6_NPose100,25,1_Focal2x --screen --project nerf__lego__CreateData_V6NPose100,25,1_Focal2x # data v6. !! This is wrong because the focal is not stored in the json file

CUDA_VISIBLE_DEVICES=0 python run_nerf_create_data.py --create_data rand --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 5000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v7_Rand_Origins_Dirs_4096RaysPerNpy --screen --project nerf__lego__CreateData_v7_Rand_Origins_Dirs_4096RaysPerNpy # data v7



# ********************* NeRF realistic synthetic dataset *********************
# create data v8 (.npy) lego
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/lego.txt --teacher_ckpt Experiments/*-195444/*/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__lego__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data

# create data v8 (.npy) ship
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/ship.txt --teacher_ckpt Experiments/*-084751/*/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/ship:data/nerf_synthetic/ship_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__ship__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data

# create data v8 (.npy) mic
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/mic.txt --teacher_ckpt Experiments/*-011144/*/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/mic:data/nerf_synthetic/mic_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__mic__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data

# create data v8 (.npy) chair
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/chair.txt --teacher_ckpt Experiments/*-011641/*/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/chair:data/nerf_synthetic/chair_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__chair__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data

# create data v8 (.npy) drums
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/drums.txt --teacher_ckpt Experiments/*-011513/*/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/drums:data/nerf_synthetic/drums_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__drums__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data

# create data v8 (.npy) ficus
CUDA_VISIBLE_DEVICES=0 python run_nerf_create_data.py --create_data rand --config configs/ficus.txt --teacher_ckpt Experiments/*-170404/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/ficus:data/nerf_synthetic/ficus_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__ficus__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data

# create data v8 (.npy) materials
CUDA_VISIBLE_DEVICES=0 python run_nerf_create_data.py --create_data rand --config configs/materials.txt --teacher_ckpt Experiments/*-170616/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/materials:data/nerf_synthetic/materials_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__materials__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data

# create data v8 (.npy) hotdog
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/hotdog.txt --teacher_ckpt Experiments/*-170420/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/hotdog:data/nerf_synthetic/hotdog_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__hotdog__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data


# ********************* NeRF real image dataset *********************
# create data v8 (.npy) room
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/room.txt --teacher_ckpt Experiments/*-121931/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_llff_data/room:data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__room__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data

# create data v8 (.npy) fern
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/fern.txt --teacher_ckpt Experiments/*-121854/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_llff_data/fern:data/nerf_llff_data/fern_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__fern__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data

# ********************* Run nerf_v3.2 on real image dataset *********************
# room
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/room_noview.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 88 --datadir_kd data/nerf_llff_data/room:data/nerf_llff_data/room_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__room__S16W181D88_DPRGB_Res_NoDir_ResMLPBody_DataV8_BS98304_Hard0.2_20xBS --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8


# kd with new data lego
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v3_NPose50,20,10 --n_pose_video 20,4,3 --N_iters 600000 --N_rand 16384 --precrop_iters -1 --screen --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDMixDataV3

CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v4_NPose50,20,10 --n_pose_video 20,4,3 --N_iters 600000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,500000:0.9 --screen --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDMixDataV4

CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDMixDataV5

CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v6_NPose100,25,1_Focal2x --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --pretrained_ckpt Exp*/*-150058/weights/600000.tar --resume --test_pretrained --screen --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDMixDataV6 # data v6. !! This is wrong because the focal is not stored in the json file

CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v7_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 4 --data_mode rays --i_update_data 20000 --screen --pretrained_ckpt Exp*/*-150058/weights/600000.tar --resume --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDDataV7 --debug --i_print 2 --i_video 5 --i_testset 8 --n_pose_video 10,4,1  # data v7

CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v7_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 4 --data_mode rays --i_update_data 100000000 --perturb 0 --screen --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDDataV7_Perturb0 # data v7, perturb 0

CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v7_Rand_Origins_Dirs_4096RaysPerNpy --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 4 --data_mode rays --i_update_data 100000000 --hard_ratio 0.0625 --screen --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDDataV7_Hard0.0625 # data v7, hard rays

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 16 --netwidth 1024 --netdepth 8 --skips 4 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 20 --data_mode rays --i_update_data 100000000 --hard_ratio 0.2 --hard_mul 20 --screen --cache_ignore data --project nerfv2__lego__S16W1024D8Skip4_DPRGB_BS98304_KDDataV8_Hard0.2_20xBS_GL21e-3 --group_l2 1e-3 # data v8, group_l2 reg

# nerf_v2 + enhance cnn, width 512
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --enhance_cnn EDSR --select_pixel_mode rand_patch --project nerfv2__lego__S4W512D32Skip8,16,24_DPRGB_BS16384_KDMixDataV5_EDSR
# nerf_v2 + enhance cnn, width 512, n_sample_per_ray 1
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 1 --netwidth 512 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --enhance_cnn EDSR --select_pixel_mode rand_patch --project nerfv2__lego__S1W512D32Skip8,16,24_DPRGB_BS16384_KDMixDataV5_EDSR

CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --enhance_cnn EDSR --select_pixel_mode rand_patch --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDMixDataV5_EDSR

# freeze model, only train enhance cnn
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --enhance_cnn EDSR --select_pixel_mode rand_patch --pretrained_ckpt Exp*/*-150058/*/1090000.tar --freeze_pretrained --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDMixDataV5_EDSR_FreezeModel

# kd with new data ship
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --config configs/ship.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --datadir_kd data/nerf_synthetic/ship:data/nerf_synthetic/ship_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__ship__S4W512D64Skip16,32,48_DPRGB_BS16384_KDMixDataV5

# kd with new data ship, only train enhance cnn
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --config configs/ship.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --datadir_kd data/nerf_synthetic/ship:data/nerf_synthetic/ship_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --enhance_cnn EDSR --select_pixel_mode rand_patch --pretrained_ckpt Exp*/*-232730/*/1200000.tar --freeze_pretrained --project nerfv2__ship__S4W512D64Skip16,32,48_DPRGB_BS16384_KDMixDataV5_EDSR_FreezeModel


# kd with new data mic
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --config configs/mic.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --datadir_kd data/nerf_synthetic/mic:data/nerf_synthetic/mic_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__mic__S4W512D64Skip16,32,48_DPRGB_BS16384_KDMixDataV5

# kd with new data chair
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --config configs/chair.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --datadir_kd data/nerf_synthetic/chair:data/nerf_synthetic/chair_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__chair__S4W512D64Skip16,32,48_DPRGB_BS16384_KDMixDataV5

# kd with new data drums
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --config configs/drums.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --datadir_kd data/nerf_synthetic/drums:data/nerf_synthetic/drums_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__drums__S4W512D64Skip16,32,48_DPRGB_BS16384_KDMixDataV5


## Train: nerf_v2, fern
# first try, use teacher to create a dataset, fixed
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --config configs/fern.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd Placeholder --teacher_ckpt Exp*/*-181900/*/200000.tar --N_iters 600000 --N_rand 16384 --precrop_iters -1 --no_batching --screen --project nerfv2__fern__S4W512D32Skip8,16,24_DPRGB_BS16384_KDMixData_FixData --debug


## Test: nerf, lego
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --config configs/lego.txt --model_name nerf --pretrained_ckpt Experiments/*-195444/*/200000.tar --render_only --n_pose_video [50,4,1] --project Test_nerf_lego_pose50,4,1 --screen

# new n_pose
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --config configs/lego.txt --model_name nerf --pretrained_ckpt Experiments/*-195444/*/200000.tar --render_only --n_pose_video sample:20,fixed:-72,fixed:5 --video_tag 20ThetasPhi-72Radius5 --project Test_nerf_lego_20ThetasPhi-72Radius5 --screen --debug

CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --config configs/lego.txt --model_name nerf --pretrained_ckpt Experiments/*-195444/*/200000.tar --render_only --n_pose_video sample:20,fixed:-72,fixed:5 --trans_origin fixed --video_tag 20ThetasPhi-72Radius5_TransOriginFixed --project Test_nerf_lego_20ThetasPhi-72Radius5_TransOriginFixed --screen --debug

## Test: nerf, fern
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --config configs/fern.txt --model_name nerf --pretrained_ckpt Experiments/*-181900/*/200000.tar --render_only --render_test --debug

## Test: nerf, ship
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --config configs/ship.txt --model_name nerf --pretrained_ckpt Experiments/*-170545/*/200000.tar --render_only --project Test_nerf_ship --screen

## Test: nerf_v2, lego
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --config configs/lego.txt --render_only --n_sample_per_ray 16 --netwidth 1024 --netdepth 8 --skips 4 --directly_predict_rgb --n_pose_video 50,4,1 --pretrained_ckpt Experiments/*-011439/weights/600000.tar --project Test_nerfv2_lego_190401_iter600000 --screen

CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --config configs/lego.txt --render_only --n_sample_per_ray 16 --netwidth 1024 --netdepth 8 --skips 4 --directly_predict_rgb --pretrained_ckpt Experiments/*-110211/weights/1200000.tar --project Video_nerfv2_lego_110211_iter1200000_pose100 --n_pose_video 100 --screen


# test changing H, W
CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --config configs/lego.txt --render_only --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --n_pose_video 10 --pretrained_ckpt Experiments/*-190401/weights/600000.tar --project Test_nerfv2_lego_190401_iter600000_HW800 --screen

CUDA_VISIBLE_DEVICES=3 python run_nerf_raybased.py --config configs/lego.txt --render_only --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --n_pose_video 10 --pretrained_ckpt Experiments/*-190401/weights/600000.tar --project Test_nerfv2_lego_190401_iter600000_HW800 --screen

# test with teacher for error map
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --config configs/lego.txt --render_only --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --n_pose_video 50,4,1 --pretrained_ckpt Experiments/*-150058/*/1000000.tar --teacher_ckpt Exp*/*-195444/*/200000.tar --project Test_nerfv2_lego_150058_iter1000000_pose50,4,1 --screen

## Test: nerf_v2, ship
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --config configs/ship.txt --render_only --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --n_pose_video 2,2,1 --pretrained_ckpt Experiments/*-232730/*/1200000.tar --teacher_ckpt Exp*/*-084751/*/200000.tar --debug

## Test: nerf_v2, fern
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --config configs/fern.txt --render_only --n_sample_per_ray 4 --netwidth 512 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --pretrained_ckpt Experiments/*-094802/weights/140000.tar --project Test_nerfv2_fern_094842_iter140000 --screen

## nerf_v5 test snippet for checking speed
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf_v5 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 1024 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__lego__S16W1024D11_DPRGB_BS98304_KDDataV8_Hard0.2_20xBS_Res_NoDir_L5 --cache_ignore data --screen --debug

## use benchmark to check speed for nerf_v3.2
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 1024 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__lego__S16W1024D11_DPRGB_BS98304_KDDataV8_Hard0.2_20xBS_Res_NoDir --cache_ignore data --screen --pretrained_ckpt Experiments/nerfv3.2__lego__S16W1024D11_DPRGB_BS98304_KDDataV8_Hard0.2_20xBS_Res_NoDir_SERVER-20210828-141538/weights/010000.tar --debug --benchmark

## convert to onnx
python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 1024 --netdepth 11 --use_residual --cache_ignore data --screen --pretrained_ckpt Exp*/*-141538/weights/1000000.tar --convert_to_onnx

## create v10 data
CUDA_VISIBLE_DEVICES=0 python create_data.py --create_data rand_images --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v10_RandOriginsDirs_10kImages --screen --project nerf__lego__CreateData_v10_RandOriginsDirs_10kImages --cache_ignore data

## nerf_v6 training
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_nerf_raybased.py --model_name nerf_v6 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 128 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v10_RandOriginsDirs_10kImages --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 32 --data_mode images_new --i_update_data 10000 --rand_crop_size 64 --use_residual --project nerfv6__lego__S16W1024D11_DPRGB_BS16ImagesCrop64_KDDataV10_Res_NoDir --cache_ignore data --screen --num_workers 24

CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf_v6 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v10_RandOriginsDirs_10kImages --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 1 --data_mode images_new --use_residual --project nerfv6__lego__S16W256D11_DPRGB_Res_NoDir__Datav10_BS1 --cache_ignore data --screen --num_workers 4 --debug --i_video 10 --i_testset 10 --i_weights 10 # use BS=1 because 400x400=160,000 ~= 98,304, to see the effect of the data dist in a batch

CUDA_VISIBLE_DEVICES=1 python create_data.py --create_data 3x3rays --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 10000000000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v11_3x3rays_queue --screen --project nerf__lego__CreateData_v11_3x3rays_queue --cache_ignore data

## nerf_v3.2 on SERVER138
CUDA_VISIBLE_DEVICES=1,0 python3.8 run_nerf_raybased.py --model_name nerf_v3.2 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 1024 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 5 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --project nerfv3.2__lego__S16W1024D11_DPRGB_BS24576_KDDataV8_Hard0.2_20xBS_Res_NoDir --cache_ignore data --screen

CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 39 --data_mode rays --use_residual --project nerfv3.2__lego__S16W256D11_DPRGB_Res_NoDir__DataV8_BS159744 --cache_ignore data --screen --debug --i_video 10 --i_testset 10 --i_weights 10 # BS159744 ~= 160,000. This exp is to run a baseline of width256 for reference.


## nerf_v3.5 on SERVER138
CUDA_VISIBLE_DEVICES=0 python3.8 run_nerf_raybased.py --model_name nerf_v3.5 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 1024 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v10_RandOriginsDirs_10kImages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 8 --data_mode images_new --rand_crop_size 64 --project nerfv3.5__lego__S16W1024D11_DPRGB_BS24576_KDDataV10_NoDir --cache_ignore data --screen --debug

## nerf_v3.4.2 on SERVER138
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf_v3.4.2 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 1024 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v11_3x3rays_queue --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 1024 --data_mode rays --use_residual --scale 3 --project nerfv3.4.2__lego__S16W1024D11_KDDataV11_Res --cache_ignore data,Experiments2,Experiments --screen --N_rand 12 --num_workers 16 --dim_dir 27 --dim_rgb 27 --i_update_data 10000

## nerf_v3.6 on SERVER138
CUDA_VISIBLE_DEVICES=0 python create_data.py --create_data 3x3rays --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 10000000000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v11_3x3rays_queue_NoRandFocal --screen --project nerf__lego__CreateData_v11_3x3rays_queue_NoRandFocal --cache_ignore data,Experiments,Experiments2,torchsearchsorted --no_rand_focal

CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf_v3.6 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 1017 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v11_3x3rays_queue --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 1024 --data_mode rays --use_residual --scale 3 --cache_ignore data,Experiments2,Experiments --screen --N_rand 6 --num_workers 16 --dim_dir 27 --dim_rgb 27 --i_update_data 10000 --project nerfv3.6__lego__S16W1017D11_KDDataV11_BS24576_Res

## nerf_v3.7 on SERVER138
CUDA_VISIBLE_DEVICES=1,2,3 python run_nerf_raybased.py --model_name nerf_v3.7 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 512 --layerwise_netwidths2 256,256,256,256,256,256,256,256,256,256,3 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v11_3x3rays_queue_NoRandFocal --n_pose_video 20,1,1 --N_iters 1200000 --data_mode rays --use_residual --scale 3 --cache_ignore data,Experiments2,Experiments --screen --N_rand 6 --num_workers 16 --dim_dir 27 --dim_rgb 27 --i_update_data 10000 --project nerfv3.7__lego__S16W512,256D11_KDDataV11_BS24576_Res

CUDA_VISIBLE_DEVICES=1,2,3 python run_nerf_raybased.py --model_name nerf_v3.7 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 1024 --layerwise_netwidths2 512,512,512,512,512,512,512,512,512,512,3 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v11_3x3rays_queue_NoRandFocal --n_pose_video 20,1,1 --N_iters 1200000 --data_mode rays --use_residual --scale 3 --cache_ignore data,Experiments2,Experiments --screen --N_rand 6 --num_workers 16 --dim_dir 27 --dim_rgb 27 --i_update_data 10000 --project nerfv3.7__lego__S16W1024,512D11_KDDataV11_BS24576_Res --debug

## create data v11
CUDA_VISIBLE_DEVICES=0 python create_data.py --create_data 3x3rays --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v11_3x3rays_queue_NoRandFocal_10000Images --screen --project nerf__lego__CreateData_v11_3x3rays_queue_NoRandFocal_10000Images --cache_ignore data,Experiments,Experiments2,torchsearchsorted --no_rand_focal --max_save 10000000000

## create data v12 and corresponding snippet to run experiments
CUDA_VISIBLE_DEVICES=0 python create_data.py --create_data 16x16patches_v2 --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v12_16x16patchesv2_10kimages --screen --project nerf__lego__CreateData_v12_16x16patchesv2_10kimages --cache_ignore data,Experiments,Experiments2

CUDA_VISIBLE_DEVICES=1,2,3 python run_nerf_raybased.py --model_name nerf_v3.8 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v12_16x16patchesv2_10kimages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 512 --data_mode 16x16patches_v2 --cache_ignore data,Experiments,Experiments2 --num_workers 12 --use_residual --i_update_data 10000 --screen --project nerfv3.8__lego__S16W256D11_DPRGB_BS512*16*16_KDDataV12_NoDir_Res

CUDA_VISIBLE_DEVICES=1,2,3 python run_nerf_raybased.py --model_name nerf_v3.8 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v12_16x16patchesv2_10kimages --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 512 --data_mode 16x16patches_v2 --cache_ignore data,Experiments,Experiments2 --num_workers 12 --use_residual --i_update_data 10000 --screen --project nerfv3.8__lego__S16W256D11_DPRGB_BS512*16*16_KDDataV12_NoDir_Res_EDSR --enhance_cnn EDSR --debug --i_video 10 --i_testset 10 --i_weights 10 # Add EDSR enhance cnn to nerf_v3.8

## create data v13 and corresponding snippet to run experiments
CUDA_VISIBLE_DEVICES=0 python create_data.py --create_data 16x16patches_v3 --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v13_16x16patchesv3_10kimages_splitsize32 --screen --project nerf__lego__CreateData_v13_16x16patchesv3_10kimages_splitsize32 --cache_ignore data,Experiments,Experiments2 --max_save 100000000000

CUDA_VISIBLE_DEVICES=1,2,3 python run_nerf_raybased.py --model_name nerf_v6 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 11 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v13_16x16patchesv3_10kimages_splitsize32 --data_mode 16x16patches_v3 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 96 --use_residual --project nerfv6__lego__S16W256D11_DPRGB_Res_NoDir__Datav13_BS3072*16*16 --cache_ignore data --screen --num_workers 12 --i_update_data 10000

## Design an EDSR-style network (i.e., using resblock in the body)
CUDA_VISIBLE_DEVICES=1,2,3 python run_nerf_raybased.py --model_name nerf_v6 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 11 --body_arch resblock --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v13_16x16patchesv3_10kimages_splitsize32 --data_mode 16x16patches_v3 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 96 --use_residual --project nerfv6__lego__S16W256D11_DPRGB_Res_NoDir__Datav13_BS3072*16*16 --cache_ignore data --screen --num_workers 12 --i_update_data 10000  --debug --i_video 10 --i_weights 10

## EDSR-style network + Data v10 (400*400)
CUDA_VISIBLE_DEVICES=0,1 python run_nerf_raybased.py --model_name nerf_v6 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 11 --body_arch resblock --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v10_RandOriginsDirs_10kImages --data_mode images_new --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 3 --use_residual --project nerfv6__lego__S16W256D11_DPRGB_Res_NoDir__Datav10_BS3x400x400 --cache_ignore data --screen --num_workers 12 --debug --i_video 10 --i_weights 10

## nerf_v6_enhance + Data v13
CUDA_VISIBLE_DEVICES=0,1 python run_nerf_raybased.py --model_name nerf_v6_enhance --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 11 --enhance_n_resblock 10 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v13_16x16patchesv3_10kimages_splitsize32 --data_mode 16x16patches_v3 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 64 --use_residual --project nerfv6enhance__lego__S16W256D11ERB10_DPRGB_Res_NoDir__Datav13_BS2048*16*16 --cache_ignore data --screen --num_workers 12 --i_update_data 10000 --debug --i_video 10 --i_weights 10 --N_rand 16

## nerf_v7 + Data v13
CUDA_VISIBLE_DEVICES=1,2,3,0 python run_nerf_raybased.py --model_name nerf_v7 --config configs/lego_noview.txt --n_sample_per_ray 16 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v13_16x16patchesv3_10kimages_splitsize32 --data_mode 16x16patches_v3 --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 96 --project nerfv7__lego__S16_DPRGB_NoDir__Datav13_BS3072*16*16 --cache_ignore data --screen --num_workers 12

## nerf_v7 + Data v10 (400*400)
CUDA_VISIBLE_DEVICES=1,0 python run_nerf_raybased.py --model_name nerf_v7 --config configs/lego_noview.txt --n_sample_per_ray 16 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v10_RandOriginsDirs_10kImages --data_mode images_new --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 3 --project nerfv7__lego__S16_DPRGB_NoDir__Datav13_BS3*400*400 --cache_ignore data --screen --num_workers 12

## nerf_v7 + Data v13 (downsample=2)
CUDA_VISIBLE_DEVICES=1,2,3,0 python run_nerf_raybased.py --model_name nerf_v7 --unet.ON --unet.n_downsample 2 --config configs/lego_noview.txt --n_sample_per_ray 16 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v13_16x16patchesv3_10kimages_splitsize32 --data_mode 16x16patches_v3 --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 96 --project nerfv7__lego__S16_DPRGB_NoDir_Down2__Datav13_BS3072*16*16 --cache_ignore data --screen --num_workers 12

## nerf_v7 + Data v10 (400*400) (downsample=2)
CUDA_VISIBLE_DEVICES=1,0 python run_nerf_raybased.py --model_name nerf_v7 --unet.ON --unet.n_downsample 2 --config configs/lego_noview.txt --n_sample_per_ray 16 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v10_RandOriginsDirs_10kImages --data_mode images_new --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 3 --project nerfv7__lego__S16_DPRGB_NoDir_Down2__Datav13_BS3x400x400 --cache_ignore data --screen --num_workers 12