import os
import numpy as np
import sys

'''usage: python <this_file> <log_path>
'''
f = sys.argv[1]

total_time = []
start_collecting = False
ours = False
num_frames = 0
for line in open(f):
    line = line.strip()
    if line.startswith("rendering all images (test)") and "rendering all images (test): 100%|" not in line: # for DONERF code
        start_collecting = True
        num_frames += 1
    
    if "Rendering video..." in line: # for our code
        start_collecting = True
        num_frames = int(line.split('n_pose: ')[1].split(")")[0])
        ours = True

    if start_collecting:
        if "[model 1] 03" in line and line.endswith("-- after inference_dict"): # for DONERF code
            t = line.split("[model 1] 03 ")[1].split(" ")[0].split('s')[0]
            total_time += [float(t)]

        if "frame, rendering done, time for this frame" in line: # for our code
            t = line.split('time for this frame: ')[1].split('s')[0]
            total_time += [float(t)]

total_time = np.array(total_time)
if ours:
    num_frames = 60
    total_time = total_time[:num_frames] # skip the first few speed data for warmup
print(f'{num_frames} frames in total. Avg. speed per frame: {total_time.sum() / num_frames:.4f}s')
    
    
