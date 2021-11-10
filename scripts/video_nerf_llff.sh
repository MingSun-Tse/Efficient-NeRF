# room
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/room.txt --pretrained_ckpt Experiments/nerf__room_S*/weights/ckpt.tar --project Video__nerf__room --cache_ignore data --render_only --n_pose_video 120 --screen

# fern
# CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/fern.txt --pretrained_ckpt Experiments/nerf__fern_S*/weights/ckpt.tar --project Video__nerf__fern --cache_ignore data --render_only --n_pose_video 120 --screen

# leaves
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/leaves.txt --pretrained_ckpt Experiments/nerf__leaves_S*/weights/ckpt.tar --project Video__nerf__leaves --cache_ignore data --render_only --n_pose_video 120 --screen

# fortress
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/fortress.txt --pretrained_ckpt Experiments/nerf__fortress_S*/weights/ckpt.tar --project Video__nerf__fortress --cache_ignore data --render_only --n_pose_video 120 --screen

# orchids
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/orchids.txt --pretrained_ckpt Experiments/nerf__orchids_S*/weights/ckpt.tar --project Video__nerf__orchids --cache_ignore data --render_only --n_pose_video 120 --screen

# flower
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/flower.txt --pretrained_ckpt Experiments/nerf__flower_S*/weights/ckpt.tar --project Video__nerf__flower --cache_ignore data --render_only --n_pose_video 120 --screen

# trex
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/trex.txt --pretrained_ckpt Experiments/nerf__trex_S*/weights/ckpt.tar --project Video__nerf__trex --cache_ignore data --render_only --n_pose_video 120 --screen

# horns
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/horns.txt --pretrained_ckpt Experiments/nerf__horns_S*/weights/ckpt.tar --project Video__nerf__horns --cache_ignore data --render_only --n_pose_video 120 --screen