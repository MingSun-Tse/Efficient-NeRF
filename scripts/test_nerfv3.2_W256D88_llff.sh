# room
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/room_noview.txt --pretrained_ckpt Experiments/nerfv3.2__room__S16W256D88*SERVER-202111*/weights/ckpt.tar --project Test__nerfv3.2__room --cache_ignore data --render_only --render_test --screen

# fern
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/fern_noview.txt --pretrained_ckpt Experiments/nerfv3.2__fern__S16W256D88*SERVER-202111*/weights/ckpt.tar --project Test__nerfv3.2__fern --cache_ignore data --render_only --render_test --screen

# leaves
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/leaves_noview.txt --pretrained_ckpt Experiments/nerfv3.2__leaves__S16W256D88*SERVER-202111*/weights/ckpt.tar --project Test__nerfv3.2__leaves --cache_ignore data --render_only --render_test --screen

# fortress
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/fortress_noview.txt --pretrained_ckpt Experiments/nerfv3.2__fortress__S16W256D88*SERVER-202111*/weights/ckpt.tar --project Test__nerfv3.2__fortress --cache_ignore data --render_only --render_test --screen

# orchids
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/orchids_noview.txt --pretrained_ckpt Experiments/nerfv3.2__orchids__S16W256D88*SERVER-202111*/weights/ckpt.tar --project Test__nerfv3.2__orchids --cache_ignore data --render_only --render_test --screen

# flower
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/flower_noview.txt --pretrained_ckpt Experiments/nerfv3.2__flower__S16W256D88*SERVER-202111*/weights/ckpt.tar --project Test__nerfv3.2__flower --cache_ignore data --render_only --render_test --screen

# trex
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/trex_noview.txt --pretrained_ckpt Experiments/nerfv3.2__trex__S16W256D88*SERVER-202111*/weights/ckpt.tar --project Test__nerfv3.2__trex --cache_ignore data --render_only --render_test --screen

# horns
CUDA_VISIBLE_DEVICES=7 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/horns_noview.txt --pretrained_ckpt Experiments/nerfv3.2__horns__S16W256D88*SERVER-202111*/weights/ckpt.tar --project Test__nerfv3.2__horns --cache_ignore data --render_only --render_test --screen