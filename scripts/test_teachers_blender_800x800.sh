# chair
CUDA_VISIBLE_DEVICES=0 python3 run_nerf_raybased.py --model_name nerf --config configs/chair_800x800.txt --pretrained_ckpt Experiments/nerf__*800x800*-031936/*/260000.tar --project Test__nerf__chair_800x800_Iter250000 --cache_ignore data,__pycache__,torchsearchsorted,imgs --render_only --render_test --testskip 1

# drums
CUDA_VISIBLE_DEVICES=1 python3 run_nerf_raybased.py --model_name nerf --config configs/drums_800x800.txt --pretrained_ckpt Experiments/nerf__*800x800*-032154/*/260000.tar --project Test__nerf__drums_800x800_Iter250000 --cache_ignore data,__pycache__,torchsearchsorted,imgs --render_only --render_test --testskip 1

# ficus
CUDA_VISIBLE_DEVICES=2 python3 run_nerf_raybased.py --model_name nerf --config configs/ficus_800x800.txt --pretrained_ckpt Experiments/nerf__*800x800*-032253/*/260000.tar --project Test__nerf__ficus_800x800_Iter250000 --cache_ignore data,__pycache__,torchsearchsorted,imgs --render_only --render_test --testskip 1

# hotdog
CUDA_VISIBLE_DEVICES=3 python3 run_nerf_raybased.py --model_name nerf --config configs/hotdog_800x800.txt --pretrained_ckpt Experiments/nerf__*800x800*-032346/*/260000.tar --project Test__nerf__hotdog_800x800_Iter250000 --cache_ignore data,__pycache__,torchsearchsorted,imgs --render_only --render_test --testskip 1

# lego
CUDA_VISIBLE_DEVICES=4 python3 run_nerf_raybased.py --model_name nerf --config configs/lego_800x800.txt --pretrained_ckpt Experiments/nerf__*800x800*-032412/*/260000.tar --project Test__nerf__lego_800x800_Iter250000 --cache_ignore data,__pycache__,torchsearchsorted,imgs --render_only --render_test --testskip 1

# materials
CUDA_VISIBLE_DEVICES=5 python3 run_nerf_raybased.py --model_name nerf --config configs/materials_800x800.txt --pretrained_ckpt Experiments/nerf__*800x800*-032424/*/260000.tar --project Test__nerf__materials_800x800_Iter250000 --cache_ignore data,__pycache__,torchsearchsorted,imgs --render_only --render_test --testskip 1

# mic
CUDA_VISIBLE_DEVICES=6 python3 run_nerf_raybased.py --model_name nerf --config configs/mic_800x800.txt --pretrained_ckpt Experiments/nerf__*800x800*-032428/*/260000.tar --project Test__nerf__mic_800x800_Iter250000 --cache_ignore data,__pycache__,torchsearchsorted,imgs --render_only --render_test --testskip 1

# ship
CUDA_VISIBLE_DEVICES=7 python3 run_nerf_raybased.py --model_name nerf --config configs/ship_800x800.txt --pretrained_ckpt Experiments/nerf__*800x800*-032434/*/260000.tar --project Test__nerf__ship_800x800_Iter250000 --cache_ignore data,__pycache__,torchsearchsorted,imgs --render_only --render_test --testskip 1
