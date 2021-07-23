## Train: nerf, lego
CUDA_VISIBLE_DEVICES=1 python run_nerf.py --config configs/lego.txt --project nerf__lego --i_print 10 --i_weights 10 --i_video 10 --debug
# make nerf_v2 back-compatible
CUDA_VISIBLE_DEVICES=2 python run_nerf_v2.py --config configs/lego.txt --model_name nerf --project nerf__lego --i_print 10 --i_weights 10 --i_video 10 --debug
# better nerf
CUDA_VISIBLE_DEVICES=3 python run_nerf_v2.py --config configs/lego.txt --model_name nerf --N_importance 256 --N_samples 128 --project nerf__lego__NSample128+256 --screen

## Train: nerf, ship
CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/ship.txt --project nerf__ship --screen --debug

## Train: nerf_v2, lego
CUDA_VISIBLE_DEVICES=1 python run_nerf_v2.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 100 --kd_with_render_pose_mode all_render_pose --kd_poses_update 10000 --n_pose_video 100 --video_poses_perturb --N_iters 400000 --N_rand 16384 --precrop_iters -1 --pretrained_ckpt Experiments/nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_KDWRenderPose100All_BS16384_SERVER142-20210704-150540/*/200000.tar --resume --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_KDWRenderPose100All_BS16384__Resume150540Iter400000

CUDA_VISIBLE_DEVICES=1 nohup python run_nerf_v2.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 100,25 --kd_with_render_pose_mode all_render_pose --n_pose_video 40,3 --video_poses_perturb --N_iters 600000 --N_rand 16384 --precrop_iters -1 --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_KDWRenderPose100,25All_BS16384 > /dev/null &

CUDA_VISIBLE_DEVICES=2 nohup python run_nerf_v2.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 20,5 --kd_with_render_pose_mode all_render_pose --n_pose_video 40,3 --video_poses_perturb --N_iters 600000 --N_rand 16384 --precrop_iters -1 --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_KDWRenderPose20,5All_BS16384 > /dev/null &

# create data (.npy) lego
CUDA_VISIBLE_DEVICES=2 python run_nerf_create_data.py --config configs/lego.txt --teacher_ckpt Experiments/nerf__lego_SERVER-20210613-195444/blender_paper_lego/200000.tar --n_pose_kd 100,25,1 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v5_NPose100,25,1 --screen --project nerf__lego__CreateData_NPose100,25,1

# create data (.npy) ship
CUDA_VISIBLE_DEVICES=0 python run_nerf_create_data.py --config configs/ship.txt --teacher_ckpt Experiments/*-084751/*/200000.tar --n_pose_kd 100,25,1 --datadir_kd data/nerf_synthetic/ship:data/nerf_synthetic/ship_v5_NPose100,25,1 --screen --project nerf__ship__CreateData_NPose100,25,1

# create data (.npy) mic
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --config configs/mic.txt --teacher_ckpt Experiments/*-011144/*/200000.tar --n_pose_kd 100,25,1 --datadir_kd data/nerf_synthetic/mic:data/nerf_synthetic/mic_v5_NPose100,25,1 --screen --project nerf__mic__CreateData_NPose100,25,1

# create data (.npy) chair
CUDA_VISIBLE_DEVICES=2 python run_nerf_create_data.py --config configs/chair.txt --teacher_ckpt Experiments/*-011641/*/200000.tar --n_pose_kd 100,25,1 --datadir_kd data/nerf_synthetic/chair:data/nerf_synthetic/chair_v5_NPose100,25,1 --screen --project nerf__chair__CreateData_NPose100,25,1

# create data (.npy) drums
CUDA_VISIBLE_DEVICES=3 python run_nerf_create_data.py --config configs/drums.txt --teacher_ckpt Experiments/*-011513/*/200000.tar --n_pose_kd 100,25,1 --datadir_kd data/nerf_synthetic/drums:data/nerf_synthetic/drums_v5_NPose100,25,1 --screen --project nerf__drums__CreateData_NPose100,25,1



# kd with new data lego
CUDA_VISIBLE_DEVICES=1 python run_nerf_v2.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v3_NPose50,20,10 --n_pose_video 20,4,3 --N_iters 600000 --N_rand 16384 --precrop_iters -1 --screen --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDMixDataV3

CUDA_VISIBLE_DEVICES=2 python run_nerf_v2.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v4_NPose50,20,10 --n_pose_video 20,4,3 --N_iters 600000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,500000:0.9 --screen --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDMixDataV4

CUDA_VISIBLE_DEVICES=0 python run_nerf_v2.py --config configs/lego.txt --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__lego__S4W1024D32Skip8,16,24_DPRGB_BS16384_KDMixDataV5


# kd with new data ship
CUDA_VISIBLE_DEVICES=0 python run_nerf_v2.py --config configs/ship.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --datadir_kd data/nerf_synthetic/ship:data/nerf_synthetic/ship_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__ship__S4W512D64Skip16,32,48_DPRGB_BS16384_KDMixDataV5

# kd with new data mic
CUDA_VISIBLE_DEVICES=1 python run_nerf_v2.py --config configs/mic.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --datadir_kd data/nerf_synthetic/mic:data/nerf_synthetic/mic_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__mic__S4W512D64Skip16,32,48_DPRGB_BS16384_KDMixDataV5

# kd with new data chair
CUDA_VISIBLE_DEVICES=2 python run_nerf_v2.py --config configs/chair.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --datadir_kd data/nerf_synthetic/chair:data/nerf_synthetic/chair_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__chair__S4W512D64Skip16,32,48_DPRGB_BS16384_KDMixDataV5

# kd with new data drums
CUDA_VISIBLE_DEVICES=3 python run_nerf_v2.py --config configs/drums.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 64 --skips 16,32,48 --directly_predict_rgb --datadir_kd data/nerf_synthetic/drums:data/nerf_synthetic/drums_v5_NPose100,25,1 --n_pose_video 50,4,1 --N_iters 1200000 --N_rand 16384 --precrop_iters -1 --i_update_data 1000 --pseudo_ratio_schedule 0:0.2,200000:0.8 --screen --project nerfv2__drums__S4W512D64Skip16,32,48_DPRGB_BS16384_KDMixDataV5




## Train: nerf_v2, fern
# first try, use teacher to create a dataset, fixed
CUDA_VISIBLE_DEVICES=3 python run_nerf_v2.py --config configs/fern.txt --n_sample_per_ray 4 --netwidth 512 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --datadir_kd Placeholder --teacher_ckpt Exp*/*-181900/*/200000.tar --N_iters 600000 --N_rand 16384 --precrop_iters -1 --no_batching --screen --project nerfv2__fern__S4W512D32Skip8,16,24_DPRGB_BS16384_KDMixData_FixData --debug


## Test: nerf, lego
# new n_pose
CUDA_VISIBLE_DEVICES=2 python run_nerf_v2.py --config configs/lego.txt --model_name nerf --pretrained_ckpt Experiments/*-195444/*/200000.tar --render_only --n_pose_video sample:20,fixed:-72,fixed:5 --video_tag 20ThetasPhi-72Radius5 --project Test_nerf_lego_20ThetasPhi-72Radius5 --screen --debug

CUDA_VISIBLE_DEVICES=2 python run_nerf_v2.py --config configs/lego.txt --model_name nerf --pretrained_ckpt Experiments/*-195444/*/200000.tar --render_only --n_pose_video sample:20,fixed:-72,fixed:5 --trans_origin fixed --video_tag 20ThetasPhi-72Radius5_TransOriginFixed --project Test_nerf_lego_20ThetasPhi-72Radius5_TransOriginFixed --screen --debug 

## Test: nerf, fern
CUDA_VISIBLE_DEVICES=3 python run_nerf_v2.py --config configs/fern.txt --model_name nerf --pretrained_ckpt Experiments/*-181900/*/200000.tar --render_only --render_test --debug

## Test: nerf, ship
CUDA_VISIBLE_DEVICES=3 python run_nerf_v2.py --config configs/ship.txt --model_name nerf --pretrained_ckpt Experiments/*-170545/*/200000.tar --render_only --project Test_nerf_ship --screen

## Test: nerf_v2, lego
CUDA_VISIBLE_DEVICES=3 python run_nerf_v2.py --config configs/lego.txt --render_only --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --n_pose_video 50,5,4 --pretrained_ckpt Experiments/*-190401/weights/600000.tar --project Test_nerfv2_lego_190401_iter600000 --screen
# test changing H, W
CUDA_VISIBLE_DEVICES=3 python run_nerf_v2.py --config configs/lego.txt --render_only --n_sample_per_ray 4 --netwidth 1024 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --n_pose_video 10 --pretrained_ckpt Experiments/*-190401/weights/600000.tar --project Test_nerfv2_lego_190401_iter600000_HW800 --screen


## Test: nerf_v2, fern
CUDA_VISIBLE_DEVICES=1 python run_nerf_v2.py --config configs/fern.txt --render_only --n_sample_per_ray 4 --netwidth 512 --netdepth 32 --skips 8,16,24 --directly_predict_rgb --pretrained_ckpt Experiments/*-094802/weights/140000.tar --project Test_nerfv2_fern_094842_iter140000 --screen
