import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter # set ytick precision
import sys, glob, os
import configargparse


parser = configargparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='test')
parser.add_argument("--exp_mark", type=str, default='')
parser.add_argument("--iter_mark", type=str, default='Iter')
parser.add_argument("--testline_mark", type=str, default='[TEST]')
parser.add_argument("--trainline_mark", type=str, default='[TRAIN]')
parser.add_argument("--max_iter", type=int, default=-1, help='the max iter in plots')
args = parser.parse_args()


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


def plot_once(expid, logf, color):
    step, test_psnr, train_hist_psnr = [], [], []
    for line in open(logf):
        if args.mode == 'test' and args.testline_mark in line:
            it = _get_value(line, 'Iter')
            if args.max_iter > 0 and it > args.max_iter:
                break
            step += [it]
            test_psnr += [_get_value(line, 'PSNR')]
            train_hist_psnr += [_get_value(line, 'Train_HistPSNR')]
        
        if args.mode == 'train' and args.trainline_mark in line:
            it = _get_value(line, 'Iter')
            if args.max_iter > 0 and it > args.max_iter:
                break
            step += [it]
            train_hist_psnr += [_get_value(line, 'hist_psnr')]
        
    
    # print(step, test_psnr, train_hist_psnr)
    if args.mode == 'test':
        ax.plot(step, test_psnr, label=f'Test_PSNR [{expid}]', color=color, linestyle='solid')
        ax.plot(step, train_hist_psnr, label=f'Train_HistPSNR [{expid}]', color=color, linestyle='dashed')
    if args.mode == 'train':
        ax.plot(step, train_hist_psnr, label=f'Train_HistPSNR [{expid}]', color=color, linestyle='dashed')

# get the path of log files
logfiles = []
for mark in args.exp_mark.split(','):
    logs = glob.glob(f'Experiments/*{mark}*/log/log.txt')
    assert len(logs) == 1
    logfiles += [logs[0]]

# main plot
colors = ['r', 'b', 'k', 'g']
ExpIDs, save_paths = [], []
for i, logf in enumerate(logfiles):
    ExpID = parse_ExpID(logf)
    ExpIDs += [ExpID]
    save_paths += [logf.replace('.txt', f'_learning_curve_{args.mode}.jpg')]
    expid = ExpID.split('-')[-1]
    plot_once(expid, logf, colors[i])

# set title with ExpIDs
title = ','.join(ExpIDs)
ax.set_title(title)
ax.grid(True, linestyle='dotted')
ax.legend()

for p in save_paths:
    fig.savefig(p)
    d = f'{os.getcwd()}/{os.path.split(p)[0]}'
    print(f'save plot to folder: {d}')


'''Usage:
py tools/plot_learning_curve.py 011145,011413 test
'''

