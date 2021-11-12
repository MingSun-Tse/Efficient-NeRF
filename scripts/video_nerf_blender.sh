
DATA="chair"
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf --cache_ignore data --config configs/${DATA}.txt --pretrained_ckpt Experiments/nerf__${DATA}_S*/*/200000.tar --n_pose_video 40 --render_only --project Video__nerf__${DATA}__NPose40 --screen

DATA="drums"
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf --cache_ignore data --config configs/${DATA}.txt --pretrained_ckpt Experiments/nerf__${DATA}_S*/*/200000.tar --n_pose_video 40 --render_only --project Video__nerf__${DATA}__NPose40 --screen

DATA="ficus"
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf --cache_ignore data --config configs/${DATA}.txt --pretrained_ckpt Experiments/nerf__${DATA}_S*/*/ckpt.tar --n_pose_video 40 --render_only --project Video__nerf__${DATA}__NPose40 --screen

DATA="hotdog"
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf --cache_ignore data --config configs/${DATA}.txt --pretrained_ckpt Experiments/nerf__${DATA}_S*/*/ckpt.tar --n_pose_video 40 --render_only --project Video__nerf__${DATA}__NPose40 --screen

DATA="lego"
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf --cache_ignore data --config configs/${DATA}.txt --pretrained_ckpt Experiments/nerf__${DATA}_S*/*/200000.tar --n_pose_video 40 --render_only --project Video__nerf__${DATA}__NPose40 --screen

DATA="materials"
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf --cache_ignore data --config configs/${DATA}.txt --pretrained_ckpt Experiments/nerf__${DATA}_S*/*/ckpt.tar --n_pose_video 40 --render_only --project Video__nerf__${DATA}__NPose40 --screen

DATA="mic"
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf --cache_ignore data --config configs/${DATA}.txt --pretrained_ckpt Experiments/nerf__${DATA}_S*/*/200000.tar --n_pose_video 40 --render_only --project Video__nerf__${DATA}__NPose40 --screen

DATA="ship"
CUDA_VISIBLE_DEVICES=1 python run_nerf_raybased.py --model_name nerf --cache_ignore data --config configs/${DATA}.txt --pretrained_ckpt Experiments/nerf__${DATA}_S*/*/200000.tar --n_pose_video 40 --render_only --project Video__nerf__${DATA}__NPose40 --screen