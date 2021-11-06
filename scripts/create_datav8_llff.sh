# create data v8 (.npy) leaves
CUDA_VISIBLE_DEVICES=0 python run_nerf_create_data.py --create_data rand --config configs/leaves.txt --teacher_ckpt Experiments/*-125237/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_llff_data/leaves:data/nerf_llff_data/leaves_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__leaves__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data --no_rand_focal --rm_existing_data

# create data v8 (.npy) fortress
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/fortress.txt --teacher_ckpt Experiments/*-125239/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_llff_data/fortress:data/nerf_llff_data/fortress_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__fortress__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data --no_rand_focal --rm_existing_data

# create data v8 (.npy) orchids
CUDA_VISIBLE_DEVICES=2 python run_nerf_create_data.py --create_data rand --config configs/orchids.txt --teacher_ckpt Experiments/*-122721/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_llff_data/orchids:data/nerf_llff_data/orchids_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__orchids__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data --no_rand_focal --rm_existing_data

# create data v8 (.npy) flower
CUDA_VISIBLE_DEVICES=3 python run_nerf_create_data.py --create_data rand --config configs/flower.txt --teacher_ckpt Experiments/*-202337/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_llff_data/flower:data/nerf_llff_data/flower_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__flower__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data --no_rand_focal --rm_existing_data

# create data v8 (.npy) trex
CUDA_VISIBLE_DEVICES=0 python run_nerf_create_data.py --create_data rand --config configs/trex.txt --teacher_ckpt Experiments/*-202357/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_llff_data/trex:data/nerf_llff_data/trex_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__trex__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data --no_rand_focal --rm_existing_data

# create data v8 (.npy) horns
CUDA_VISIBLE_DEVICES=1 python run_nerf_create_data.py --create_data rand --config configs/horns.txt --teacher_ckpt Experiments/*-214300/weights/ckpt.tar --n_pose_kd 10000 --datadir_kd data/nerf_llff_data/horns:data/nerf_llff_data/horns_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --screen --project nerf__horns__CreateData_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --cache_ignore data --no_rand_focal --rm_existing_data