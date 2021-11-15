from PIL import Image
import os
import sys
import numpy as np


is_img = lambda x: os.path.splitext(x)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']

'''Usage: python  <this_file>  <inDir>  50,30  40,40 
'''
# ***************************************
inDir = sys.argv[1]
h0, w0 = [int(i) for i in sys.argv[2].split(',')]
cuth, cutw = [int(i) for i in sys.argv[3].split(',')]
# ***************************************

all_imgs = [f'{inDir}/{i}' for i in os.listdir(inDir) if is_img(i)]
for img_path in all_imgs:
    ext = os.path.splitext(img_path)[-1]
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    print(img.shape)
    cut = img[h0:h0+cuth, w0:w0+cutw]
    cut = Image.fromarray(cut)
    savepath = img_path.replace(ext, f'_h0{h0}_w0{w0}_cuth{cuth}_cutw{cutw}' + ext)
    cut.save(savepath)
    print(f'save to "{savepath}"')