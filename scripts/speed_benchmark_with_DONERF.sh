# ours
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 88 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 65 --N_iters 210000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --render_only --pretrained_ckpt Experiments/*-094904/weights/ckpt.tar --project SpeedBenchmark_nerfv3.2_W181D88

CUDA_VISIBLE_DEVICES="" python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 181 --netdepth 88 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --n_pose_video 65 --N_iters 210000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --cache_ignore data --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --render_only --pretrained_ckpt Experiments/*-094904/weights/ckpt.tar --project SpeedBenchmark_nerfv3.2_W181D88_CPU_SERVER005


# donerf
CUDA_VISIBLE_DEVICES=1 python test.py -c ../configs/DONeRF_16_samples.ini --data data/bulldozer --logDir PlaceHolderDONERF --project SpeedBenchmark_DONERF_16Samples --speed # note, change the inferenceChunckSize to 160000 in the DONeRF_16_samples.ini

CUDA_VISIBLE_DEVICES="" python test.py -c ../configs/DONeRF_16_samples.ini --data data/bulldozer --logDir PlaceHolderDONERF --project SpeedBenchmark_DONERF_16Samples_CPU_SERVER005 --speed


# nerf
CUDA_VISIBLE_DEVICES=1 python test.py -c ../configs/NeRF_coarse_plus_fine.ini --data data/bulldozer --logDir PlaceHolder --project SpeedBenchmark_DONERF_nerf --speed # 160000 in the NeRF_coarse_plus_fine.ini is not feasible, so use the original 8000

CUDA_VISIBLE_DEVICES="" python test.py -c ../configs/NeRF_coarse_plus_fine.ini --data data/bulldozer --logDir PlaceHolder --project SpeedBenchmark_DONERF_nerf_CPU_SERVER005 --speed



# Benchmark on SERVER005:
# DONERF
[183925 20880 2021/11/15-18:39:34] [model 1] 03 0.4276s -- after inference_dict
[183925 20880 2021/11/15-18:39:34] =======> FLOPs 7093440.023225
Generating diff and flip images:   7%|6         | 4/60 [00:02<00:40,  1.37it/s][183925 20880 2021/11/15-18:39:35] [model 0] 00 0.0000s -- after get_batch_input
[183925 20880 2021/11/15-18:39:35] [model 0] 01 0.0095s -- after model_input
[183925 20880 2021/11/15-18:39:35] [model 0] 02 0.0288s -- after model forward
[183925 20880 2021/11/15-18:39:35] [model 0] 03 0.0288s -- after inference_dict
[183925 20880 2021/11/15-18:39:35] [model 1] 00 0.0289s -- after get_batch_input
[183925 20880 2021/11/15-18:39:35] [model 1] 01 0.0836s -- after model_input
[183925 20880 2021/11/15-18:39:35] [model 1] 02 0.4263s -- after model forward
[183925 20880 2021/11/15-18:39:35] [model 1] 03 0.4279s -- after inference_dict
[183925 20880 2021/11/15-18:39:35] =======> FLOPs 7093440.023225
Generating diff and flip images:   8%|8         | 5/60 [00:03<00:39,  1.39it/s][183925 20880 2021/11/15-18:39:35] [model 0] 00 0.0000s -- after get_batch_input
[183925 20880 2021/11/15-18:39:35] [model 0] 01 0.0095s -- after model_input
[183925 20880 2021/11/15-18:39:35] [model 0] 02 0.0288s -- after model forward
[183925 20880 2021/11/15-18:39:35] [model 0] 03 0.0288s -- after inference_dict
[183925 20880 2021/11/15-18:39:35] [model 1] 00 0.0289s -- after get_batch_input
[183925 20880 2021/11/15-18:39:35] [model 1] 01 0.0834s -- after model_input
[183925 20880 2021/11/15-18:39:36] [model 1] 02 0.4266s -- after model forward
[183925 20880 2021/11/15-18:39:36] [model 1] 03 0.4281s -- after inference_dict
[183925 20880 2021/11/15-18:39:36] =======> FLOPs 7093440.023225

# Ours, W181D88
[184046 21288 2021/11/15-18:40:56] 
[184046 21288 2021/11/15-18:40:56] [#15] frame, rendering begins
[184046 21288 2021/11/15-18:40:56] [#15] frame, prepare input (embedding): 0.0191s
[184046 21288 2021/11/15-18:40:56] [#15] frame, model forward: 0.1908s
[184046 21288 2021/11/15-18:40:56] [#15] frame, rendering done, time for this frame: 0.2100s
[184046 21288 2021/11/15-18:40:56] 
[184046 21288 2021/11/15-18:40:56] [#16] frame, rendering begins
[184046 21288 2021/11/15-18:40:56] [#16] frame, prepare input (embedding): 0.0190s
[184046 21288 2021/11/15-18:40:56] [#16] frame, model forward: 0.1912s
[184046 21288 2021/11/15-18:40:56] [#16] frame, rendering done, time for this frame: 0.2102s
[184046 21288 2021/11/15-18:40:56] 
[184046 21288 2021/11/15-18:40:56] [#17] frame, rendering begins
[184046 21288 2021/11/15-18:40:57] [#17] frame, prepare input (embedding): 0.0191s
[184046 21288 2021/11/15-18:40:57] [#17] frame, model forward: 0.1910s
[184046 21288 2021/11/15-18:40:57] [#17] frame, rendering done, time for this frame: 0.2102s
