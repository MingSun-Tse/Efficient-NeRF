from PIL import Image, ImageDraw as D
from matplotlib.patches import Rectangle
import os, copy
import sys
import numpy as np
import configargparse


is_img = lambda x: os.path.splitext(x)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']

"""Usage: 
py tools/cutout_showcase.py --inDir picked_results_blender_llff --include hotdog --cut_hw 100,100 --upperleft 210,200
py tools/cutout_showcase.py --inDir picked_results_blender_800x800 --include chair --full_res --cut_hw 200,200 --upperleft 360,200 # 800x800 images

"""

parser = configargparse.ArgumentParser()
parser.add_argument("--inDir", type=str, default='')
parser.add_argument("--include", type=str, default='')
parser.add_argument("--exclude", type=str, default='')
parser.add_argument("--upperleft", type=str, default='')
parser.add_argument("--cut_hw", type=str, default='')
parser.add_argument("--plot_rect", action='store_true')
parser.add_argument("--full_res", action='store_true')
args = parser.parse_args()

def plot_rect(img, w1, h1, h, w, color='red'):
    '''img should be a PIL Image instance'''
    draw = D.Draw(img)
    draw.rectangle([(w1, h1),(w1+w, h1+h)], outline=color, width=1)
    return img

# ***************************************
inDir = args.inDir
h0, w0 = [float(i) for i in args.upperleft.split(',')]
cuth, cutw = [int(i) for i in args.cut_hw.split(',')]
include = [x.strip() for x in args.include.split(',')] if args.include else ['.']
exclude = [x.strip() for x in args.exclude.split(',')] if args.exclude else []
exclude += ['cut']
# ***************************************
all_save_paths = []

all_imgs = [f'{inDir}/{i}' for i in os.listdir(inDir) if is_img(i) and any([x in i for x in include]) and not any([x in i for x in exclude])]
print(all_imgs)
for img_path in all_imgs:
    ext = os.path.splitext(img_path)[-1]
    img = Image.open(img_path)
    
    # Resize
    if not args.full_res:
        if img.size[0] == 800 and img.size[1] == 800: # blender dataset
            # img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
            img = img.resize((400, 400))
            print(f'==> resize image to 400x400: "{img_path}"')
    
    # To white bkgd
    img = np.array(img)
    num_channels = img.shape[2]
    if num_channels == 4:
        img = img / 255.
        img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:]) # This works when the img range is in [0, 1]
        img = np.uint8(img * 255.)
        print(f'==> num_channels = {num_channels}, convert to white background')

    # Cut
    print(img.shape)
    if h0 < 1: # h0 is a ratio in (0, 1)
        h0, w0 = int(h0 * img.shape[0]), int(w0 * img.shape[1])
    else:
        h0, w0 = int(h0), int(w0)
    cut = img[h0:h0+cuth, w0:w0+cutw]
    cut = Image.fromarray(cut)
    
    # Plot rect on the GT large image
    if 'gt' in img_path:
        img_copy = Image.fromarray(img)
        draw = D.Draw(img_copy)
        draw.rectangle([(w0, h0), (w0+cutw, h0+cuth)], outline='red', width=2)
        savepath = img_path.replace(ext, f'_h0{h0}_w0{w0}_cuth{cuth}_cutw{cutw}_rect' + ext)
        img_copy.save(savepath)
        all_save_paths += [savepath]
       
    # Save
    savepath = img_path.replace(ext, f'_h0{h0}_w0{w0}_cuth{cuth}_cutw{cutw}' + ext)
    cut.save(savepath)
    print(f'save to "{savepath}"\n')
    all_save_paths += [savepath]

input_str = input("Is it okay? ")
if input_str.lower() not in ['yes', 'y']:
    for p in all_save_paths:
        os.remove(p)

