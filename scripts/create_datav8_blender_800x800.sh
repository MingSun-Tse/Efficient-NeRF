# chair
CUDA_VISIBLE_DEVICES=0 python3 run_nerf_create_data.py --create_data rand --config configs/chair_800x800.txt --teacher_ckpt Experiments/nerf__*800x800*-031936/*/260000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/chair:data/nerf_synthetic/chair_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --screen --project nerf__chair__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --cache_ignore data,__pycache__,torchsearchsorted,imgs

# drums
CUDA_VISIBLE_DEVICES=1 python3 run_nerf_create_data.py --create_data rand --config configs/drums_800x800.txt --teacher_ckpt Experiments/nerf__*800x800*-032154/*/260000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/drums:data/nerf_synthetic/drums_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --screen --project nerf__drums__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --cache_ignore data,__pycache__,torchsearchsorted,imgs

# ficus
CUDA_VISIBLE_DEVICES=2 python3 run_nerf_create_data.py --create_data rand --config configs/ficus_800x800.txt --teacher_ckpt Experiments/nerf__*800x800*-032253/*/260000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/ficus:data/nerf_synthetic/ficus_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --screen --project nerf__ficus__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --cache_ignore data,__pycache__,torchsearchsorted,imgs

# hotdog
CUDA_VISIBLE_DEVICES=3 python3 run_nerf_create_data.py --create_data rand --config configs/hotdog_800x800.txt --teacher_ckpt Experiments/nerf__*800x800*-032346/*/260000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/hotdog:data/nerf_synthetic/hotdog_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --screen --project nerf__hotdog__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --cache_ignore data,__pycache__,torchsearchsorted,imgs

# lego
CUDA_VISIBLE_DEVICES=4 python3 run_nerf_create_data.py --create_data rand --config configs/lego_800x800.txt --teacher_ckpt Experiments/nerf__*800x800*-032412/*/260000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --screen --project nerf__lego__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --cache_ignore data,__pycache__,torchsearchsorted,imgs

# materials
CUDA_VISIBLE_DEVICES=5 python3 run_nerf_create_data.py --create_data rand --config configs/materials_800x800.txt --teacher_ckpt Experiments/nerf__*800x800*-032424/*/260000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/materials:data/nerf_synthetic/materials_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --screen --project nerf__materials__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --cache_ignore data,__pycache__,torchsearchsorted,imgs

# mic
CUDA_VISIBLE_DEVICES=6 python3 run_nerf_create_data.py --create_data rand --config configs/mic_800x800.txt --teacher_ckpt Experiments/nerf__*800x800*-032428/*/260000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/mic:data/nerf_synthetic/mic_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --screen --project nerf__mic__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --cache_ignore data,__pycache__,torchsearchsorted,imgs

# ship
CUDA_VISIBLE_DEVICES=7 python3 run_nerf_create_data.py --create_data rand --config configs/ship_800x800.txt --teacher_ckpt Experiments/nerf__*800x800*-032434/*/260000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/ship:data/nerf_synthetic/ship_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --screen --project nerf__ship__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages_800x800 --cache_ignore data,__pycache__,torchsearchsorted,imgs
