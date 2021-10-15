import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter # set ytick precision
import sys, glob

def parse_ExpID(path):
    '''parse out the ExpID from 'path', which can be a file or directory.
    Example: Experiments/AE__ckpt_epoch_240.pth__LR1.5__originallabel__vgg13_SERVER138-20200829-202307/gen_img
    Example: Experiments/AE__ckpt_epoch_240.pth__LR1.5__originallabel__vgg13_SERVER-20200829-202307/gen_img
    '''
    return 'SERVER' + path.split('_SERVER')[1].split('/')[0]


def _get_value(line, key, type_func=float, exact_key=True):
    if exact_key: # back compatibility
        value = line.split(key)[1].strip().split()[0]
        if value.endswith(')'): # hand-fix case: "Epoch 23)"
            value = value[:-1]
        value = type_func(value)
    else:
        line_seg = line.split()
        for i in range(len(line_seg)):
            if key in line_seg[i]: # example: 'Acc1: 0.7'
                break
        if i == len(line_seg) - 1:
            return None # did not find the <key> in this line
        value = type_func(line_seg[i + 1])
    return value


# create a fig
figsize = (6, 4)
fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)

def plot_once(expid, logf, color, mode='test'):
    step, test_psnr, train_hist_psnr = [], [], []
    for line in open(logf):
        if mode == 'test' and '[TEST]' in line:
            step += [_get_value(line, 'Iter')]
            test_psnr += [_get_value(line, 'PSNR')]
            train_hist_psnr += [_get_value(line, 'Train_HistPSNR')]
        
        if mode == 'train' and '[TRAIN]' in line:
            step += [_get_value(line, 'Iter')]
            train_hist_psnr += [_get_value(line, 'hist_psnr')]
    
    # print(step, test_psnr, train_hist_psnr)
    if mode == 'test':
        ax.plot(step, test_psnr, label=f'Test_PSNR [{expid}]', color=color, linestyle='solid')
        ax.plot(step, train_hist_psnr, label=f'Train_HistPSNR [{expid}]', color=color, linestyle='dashed')
    if mode == 'train':
        ax.plot(step, train_hist_psnr, label=f'Train_HistPSNR [{expid}]', color=color, linestyle='dashed')

# get the path of log files
logfiles = []
for mark in sys.argv[1].split(','):
    logs = glob.glob(f'Experiments/*{mark}*/log/log.txt')
    assert len(logs) == 1
    logfiles += [logs[0]]

# main plot
colors = ['r', 'b', 'k', 'g']
ExpIDs, save_paths = [], []
mode = sys.argv[2]
for i, logf in enumerate(logfiles):
    ExpID = parse_ExpID(logf)
    ExpIDs += [ExpID]
    save_paths += [logf.replace('.txt', f'_learning_curve_{mode}.jpg')]
    expid = ExpID.split('-')[-1]
    plot_once(expid, logf, colors[i], mode=mode)

# set title with ExpIDs
title = ','.join(ExpIDs)
ax.set_title(title)
ax.grid(True)
ax.legend()

for p in save_paths:
    fig.savefig(p)
    print(f'save plot to {p}')


'''Usage:
py tools/plot_learning_curve.py 011145,011413 test
'''

