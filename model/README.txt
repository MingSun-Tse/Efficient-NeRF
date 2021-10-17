model evolution path:
* nerf_v2: adapted from original nerf
* nerf_v3: move ray_d to the input layer (no skip layers)
    --> nerf_v3.2: move pts encoding out
    --> nerf_v3.3: 1x1 conv, = nerf_v3.2 (also named nerf_v5)
        --> nerf_v6: use 3x3 conv in middle layers 

################ For faster (after internship exit) ################ 
* nerf_v3.4: 3x3 patch idea, input_dim*9, out_dim*9 (pure MLP) 
    --> nerf_v3.4.2: the SAME model as nerf_v3.4
        --> nerf_v3.6: 1x1 conv, use group conv, diverge
    --> nerf_v3.7: two paths (Zeng's idea: low-freq + high-freq), deprecated!
    --> nerf_v3.8: U-Net style
    
* nerf_v3.5: input 3x3 conv, use PixelShuffle -- slow, deprecated!
* nerf_v4: two rays (I proposed. Also have a figure to illustrate the idea. I think the formulation is elegant but not working)


Data evolution path:
* data v7: This is the 1st working scheme: store rays in .npy instead of storing images
* data v8: 10k images vs. 5k in v7



`unet` folder refers to the PyTorch UNet implementation at: https://github.com/milesial/Pytorch-UNet