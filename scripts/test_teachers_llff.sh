# room
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf --config configs/room.txt --pretrained_ckpt Experiments/nerf__room_S*/weights/ckpt.tar --project Test__nerf__room --cache_ignore data --render_only --render_test --screen

# fern
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf --config configs/fern.txt --pretrained_ckpt Experiments/nerf__fern_S*/weights/ckpt.tar --project Test__nerf__fern --cache_ignore data --render_only --render_test --screen

# leaves
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf --config configs/leaves.txt --pretrained_ckpt Experiments/nerf__leaves_S*/weights/ckpt.tar --project Test__nerf__leaves --cache_ignore data --render_only --render_test --screen

# fortress
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf --config configs/fortress.txt --pretrained_ckpt Experiments/nerf__fortress_S*/weights/ckpt.tar --project Test__nerf__fortress --cache_ignore data --render_only --render_test --screen

# orchids
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf --config configs/orchids.txt --pretrained_ckpt Experiments/nerf__orchids_S*/weights/ckpt.tar --project Test__nerf__orchids --cache_ignore data --render_only --render_test --screen

# flower
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf --config configs/flower.txt --pretrained_ckpt Experiments/nerf__flower_S*/weights/ckpt.tar --project Test__nerf__flower --cache_ignore data --render_only --render_test --screen

# trex
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf --config configs/trex.txt --pretrained_ckpt Experiments/nerf__trex_S*/weights/ckpt.tar --project Test__nerf__trex --cache_ignore data --render_only --render_test --screen

# horns
CUDA_VISIBLE_DEVICES=2 python run_nerf_raybased.py --model_name nerf --config configs/horns.txt --pretrained_ckpt Experiments/nerf__horns_S*/weights/ckpt.tar --project Test__nerf__horns --cache_ignore data --render_only --render_test --screen