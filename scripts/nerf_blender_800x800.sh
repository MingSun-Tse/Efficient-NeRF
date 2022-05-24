CUDA_VISIBLE_DEVICES=0 python3 run_nerf_raybased.py --model_name nerf --config configs/chair_800x800.txt --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --N_iters 300000 --i_test 50000 --i_video 100000000 --i_weights 10000 --project nerf__chair__800x800

CUDA_VISIBLE_DEVICES=1 python3 run_nerf_raybased.py --model_name nerf --config configs/drums_800x800.txt --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --N_iters 300000 --i_test 50000 --i_video 100000000 --i_weights 10000 --project nerf__drums__800x800

CUDA_VISIBLE_DEVICES=2 python3 run_nerf_raybased.py --model_name nerf --config configs/ficus_800x800.txt --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --N_iters 300000 --i_test 50000 --i_video 100000000 --i_weights 10000 --project nerf__ficus__800x800

CUDA_VISIBLE_DEVICES=3 python3 run_nerf_raybased.py --model_name nerf --config configs/hotdog_800x800.txt --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --N_iters 300000 --i_test 50000 --i_video 100000000 --i_weights 10000 --project nerf__hotdog__800x800

CUDA_VISIBLE_DEVICES=4 python3 run_nerf_raybased.py --model_name nerf --config configs/lego_800x800.txt --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --N_iters 300000 --i_test 50000 --i_video 100000000 --i_weights 10000 --project nerf__lego__800x800

CUDA_VISIBLE_DEVICES=5 python3 run_nerf_raybased.py --model_name nerf --config configs/materials_800x800.txt --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --N_iters 300000 --i_test 50000 --i_video 100000000 --i_weights 10000 --project nerf__materials__800x800

CUDA_VISIBLE_DEVICES=6 python3 run_nerf_raybased.py --model_name nerf --config configs/mic_800x800.txt --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --N_iters 300000 --i_test 50000 --i_video 100000000 --i_weights 10000 --project nerf__mic__800x800

CUDA_VISIBLE_DEVICES=7 python3 run_nerf_raybased.py --model_name nerf --config configs/ship_800x800.txt --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --N_iters 300000 --i_test 50000 --i_video 100000000 --i_weights 10000 --project nerf__ship__800x800
