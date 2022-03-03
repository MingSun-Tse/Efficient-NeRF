import os, sys
inDir1 = os.path.abspath(sys.argv[1])
inDir2 = os.path.abspath(sys.argv[2])
outDir = os.path.abspath(sys.argv[3])

def create_softlink(inDir):
    for x in os.listdir(inDir):
        x = os.path.join(inDir, x)
        if os.path.isfile(x):
            script = f'ln -s {x} .'
            os.system(script)
            print(script)

os.makedirs(outDir)
os.chdir(outDir)
create_softlink(inDir1)
create_softlink(inDir2)

