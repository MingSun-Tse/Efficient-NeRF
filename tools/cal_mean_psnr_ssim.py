import numpy as np
import sys

a = sys.argv[1]
b = [float(x) for x in a.split()]
print(f'psnr avg: {np.mean(b[0:-1:2])}')
print(f'ssim avg: {np.mean(b[1:-1:2])}')
