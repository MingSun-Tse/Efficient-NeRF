import os, sys
from multiprocessing import Pool
inDir1 = os.path.abspath(sys.argv[1])
inDir2 = os.path.abspath(sys.argv[2])
outDir = os.path.abspath(sys.argv[3])
cnt = [0]

def fn(x):
    os.symlink(x, os.path.split(x)[-1])
    cnt[0] += 1
    print(f'{cnt[0]}: {x}')

def create_softlink(inDir):
    files = [os.path.join(inDir, x) for x in os.listdir(inDir)]
    pool = Pool()
    pool.map(fn, files)

os.makedirs(outDir)
os.chdir(outDir)
create_softlink(inDir2)
create_softlink(inDir1)
