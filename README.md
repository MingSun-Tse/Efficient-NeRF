# R2L: Distilling NeRF to NeLF

This repository is for the new neral light field (NeLF) method introduced in the following ECCV'22 paper:
> **R2L: Distilling Neural Radiance Field to Neural Light Field for Efficient Novel View Synthesis [[Arxiv](https://arxiv.org/abs/2203.17261)] [[Project](https://snap-research.github.io/R2L/)]** \
> [Huan Wang](http://huanwang.tech/) [1,2], [Jian Ren](https://alanspike.github.io/) [1], [Zeng Huang](https://zeng.science/) [1], [Kyle Olszewski](https://kyleolsz.github.io/) [1], [Menglei Chai](https://mlchai.com/) [1], [Yun Fu](http://www1.ece.neu.edu/~yunfu/) [2], and [Sergey Tulyakov](http://www.stulyakov.com/) [1] \
> [1] Snap Inc. [2] Northeastern University \
> Work done when Huan was an intern at Snap Inc.

**[TLDR]** We present R2L, a deep (88-layer) residual MLP network that can represent the neural *light* field (NeLF) of complex synthetic and real-world scenes. It is featured by compact representation size (~20MB storage size), faster rendering speed (~30x speedup than NeRF), significantly improved visual quality (1.4dB boost than NeRF), with no whistles and bells (no special data structure or parallelism required).

<center><img src="frontpage.png" width="700" hspace="10"></center>


## Reproducing Our Results
### 1. Set up (original) data
```bash
sh scripts/download_data_v2.sh
```

### 2. Set up environment with Anaconda
- `conda create --name R2L python=3.9.6`
- `conda activate R2L`
- `pip install -r requirements.txt` (torch==1.9.0, torchvision==0.10.0)

### 3. Test our trained models
- Download our trained models here.
- Run
```bash
CUDA_VISIBLE_DEVICES=0 python run_nerf_raybased.py --model_name nerf_v3.2 --config configs/chair_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/chair_v8_Rand_Origins_Dirs_4096RaysPerNpy_10kImages --use_residual --cache_ignore data --trial.ON --trial.body_arch resmlp --pretrained_ckpt R2L_models/W256D88__blender_chair__400x400.tar --render_only --render_test --testskip 1 --project Test__R2L_W256D88__blender_chair__400x400
```  
Here we only show the example of scene chair. You may test on other scenes simply by changing all the `chair` word segments to other scene names.
 


### 4. Train R2L models
There are two major steps in R2L training. (1) Use *pretrained* NeRF model to generate synthetic data and train R2L network on the synthetic data -- this step can make our R2L model perform *comparably* to the NeRF teacher; (2) Finetune the R2L model in step (1) on the *real* data -- this step will further boost the performance and make our R2L model *outperform* the NeRF teacher.

The detailed step-by-step training pipeline is as follows.

Step 1. Pretrain a NeRF model (we simply follow the instructions [here](https://github.com/yenchenlin/nerf-pytorch))
```bash

```

Step 2. Use the pretrained NeRF model to generate synthetic data (saved in .npy format):
```bash

```
Step 3. Train R2L model on the synthetic data:
```bash

```

Step 4. Convert original real data (images) to our .npy format:
```bash

```

Step 5. Finetune the R2L model in Step 3 on the data in Step 4:
```bash

```
Note, this step is pretty fast and prone to overfitting, so do not finetune it too much. We simply set the finetuning steps based on our validation.


## Results

See more results and videos on our [project webpage](https://snap-research.github.io/R2L/).


## Acknowledgments
In this code we refer to the following implementations: [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), [smilelogging](https://github.com/MingSun-Tse/smilelogging). Great thanks to them! We especially thank [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). Our code largely builds upon their wonderful implementation. We also greatly thank the anounymous ECCV'22 reviewers for the constructive comments to help us improve the paper.

## Reference

If our work or code helps you, please consider citing our paper. Thank you!

    @article{wang2022r2l,
      Author = {Huan Wang and Jian Ren and Zeng Huang and Kyle Olszewski and Menglei Chai and Yun Fu and Sergey Tulyakov},
      Title = {R2L: Distilling Neural Radiance Field to Neural Light Field for Efficient Novel View Synthesis},
      Booktitle = {ECCV},
      Year = {2022}
    }




