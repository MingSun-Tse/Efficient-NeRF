# chair
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/chair.txt --pretrained_ckpt Experiments/nerf__*-011641/*/200000.tar --project Test__nerf__chair --cache_ignore data,__pycache__,torchsearchsorted,imgs  --render_only --render_test --testskip 1

# drums
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/drums.txt --pretrained_ckpt Experiments/nerf__*-011513/*/200000.tar --project Test__nerf__drums --cache_ignore data,__pycache__,torchsearchsorted,imgs  --render_only --render_test --testskip 1

# ficus
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/ficus.txt --pretrained_ckpt Experiments/nerf__*-170404/*/ckpt.tar --project Test__nerf__ficus --cache_ignore data,__pycache__,torchsearchsorted,imgs  --render_only --render_test --testskip 1

# hotdog
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/hotdog.txt --pretrained_ckpt Experiments/nerf__*-170420/*/ckpt.tar --project Test__nerf__hotdog --cache_ignore data,__pycache__,torchsearchsorted,imgs  --render_only --render_test --testskip 1

# lego
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/lego.txt --pretrained_ckpt Experiments/nerf__*-195444/*/200000.tar --project Test__nerf__lego --cache_ignore data,__pycache__,torchsearchsorted,imgs  --render_only --render_test --testskip 1

# materials
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/materials.txt --pretrained_ckpt Experiments/nerf__*-170616/*/ckpt.tar --project Test__nerf__materials --cache_ignore data,__pycache__,torchsearchsorted,imgs  --render_only --render_test --testskip 1

# mic
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/mic.txt --pretrained_ckpt Experiments/nerf__*-011144/*/200000.tar --project Test__nerf__mic --cache_ignore data,__pycache__,torchsearchsorted,imgs  --render_only --render_test --testskip 1

# ship
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf --config configs/ship.txt --pretrained_ckpt Experiments/nerf__*-084751/*/200000.tar --project Test__nerf__ship --cache_ignore data,__pycache__,torchsearchsorted,imgs  --render_only --render_test --testskip 1