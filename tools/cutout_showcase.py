from PIL import Image
import os
import sys
import numpy as np
import configargparse


is_img = lambda x: os.path.splitext(x)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']

'''Usage: py tools/cutout_showcase.py --inDir picked_results_blender --cut_hw 100,100 --upperleft 210,200  
'''

parser = configargparse.ArgumentParser()
parser.add_argument("--inDir", type=str, default='')
parser.add_argument("--include", type=str, default='')
parser.add_argument("--exclude", type=str, default='')
parser.add_argument("--upperleft", type=str, default='')
parser.add_argument("--cut_hw", type=str, default='')
args = parser.parse_args()

# ***************************************
inDir = args.inDir
h0, w0 = [int(i) for i in args.upperleft.split(',')]
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
    img = Image.open(img_path).convert('RGB')
    if img.size[0] == 800:
        img = img.resize([400, 400])
        print(f'resize image to 400x400: "{img_path}"')
    img = np.array(img)
    print(img.shape)
    cut = img[h0:h0+cuth, w0:w0+cutw]
    cut = Image.fromarray(cut)
    savepath = img_path.replace(ext, f'_h0{h0}_w0{w0}_cuth{cuth}_cutw{cutw}' + ext)
    cut.save(savepath)
    print(f'save to "{savepath}"')
    all_save_paths += [savepath]

input_str = input("Is it okay? ")
if input_str.lower() not in ['yes', 'y']:
    for p in all_save_paths:
        os.remove(p)

