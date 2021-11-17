import matplotlib.pyplot as plt
plt.style.use(['science']) # 'ieee'
import numpy as np
from matplotlib.ticker import FormatStrFormatter # set ytick precision
import sys, glob, os
import configargparse
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


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

def plot_once(expid, logf, color, label):
    step, test_psnr, train_hist_psnr = [], [], []
    for line in open(logf):
        if args.mode == 'test' and args.testline_mark in line:
            it = _get_value(line, 'Iter')
            if args.max_iter > 0 and it > args.max_iter:
                break
            step += [it]
            test_psnr += [_get_value(line, 'TestPSNRv2')]
            train_hist_psnr += [_get_value(line, 'TrainHistPSNR')]
        
        if args.mode == 'train' and args.trainline_mark in line:
            it = _get_value(line, 'Iter')
            if args.max_iter > 0 and it > args.max_iter:
                break
            step += [it]
            train_hist_psnr += [_get_value(line, 'hist_psnr')]
        
    step = np.array(step) / xunit
    if args.mode == 'test':
        ax.plot(step, test_psnr, label=f'{label} (test)', color=color, linestyle='dashed', linewidth=linewidth)
        ax.plot(step, train_hist_psnr, label=f'{label} (train)', color=color, linestyle='solid', linewidth=linewidth)
        axins.plot(step, test_psnr, color=color, linestyle='dashed', linewidth=linewidth)

# *********************************************************************
# put data here!!
args.max_iter = 200000
args.mode = 'test'
args.exp_mark = '041330,024531,024904,151044,135713,094904'
colors = ['m', 'g', 'y', 'b', 'cyan', 'r']
labels = ['S=0.1k', 'S=0.5k', 'S=1k', 'S=2.5k', 'S=5k', 'S=10k']
xunit = 1000

# other settings
legend_fs = 9
xlabel_fs = 16
ylabel_fs = 16
linewidth = 1.5
ticklabelsize = 14
savename = f'ablation_pseudo_sample_size.pdf'
# *********************************************************************

# ******************************
# Make the zoom-in plot
x1 = 175000 / xunit
x2 = 200000 / xunit
y1 = 28.5
y2 = 30
axins = zoomed_inset_axes(ax, zoom=3, loc=8)

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)
ax.tick_params(axis='both', labelsize=ticklabelsize)
# ******************************

# get the path of log files
logfiles = []
for mark in args.exp_mark.split(','):
    logs = glob.glob(f'Experiments/*{mark}*/log/log.txt')
    assert len(logs) == 1
    logfiles += [logs[0]]

# main plot
ExpIDs, save_paths = [], [f'./{savename}'] # also save to current folder for easy check
for i, logf in enumerate(logfiles):
    ExpID = parse_ExpID(logf)
    ExpIDs += [ExpID]
    save_paths += [f'{os.path.split(logf)[0]}/{savename}']
    expid = ExpID.split('-')[-1]
    plot_once(expid, logf, colors[i], labels[i])

ax.grid(True, linestyle='dashed')
ax.legend(fontsize=legend_fs)
ax.set_xlabel('Iteration (k)', fontsize=xlabel_fs)
ax.set_ylabel('PSNR (dB)', fontsize=ylabel_fs)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="k", linewidth=0.4, linestyle='dashed')
# see https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.inset_locator.mark_inset.html 

for p in save_paths:
    fig.savefig(p, bbox_inches='tight')
    d = f'{os.getcwd()}/{os.path.split(p)[0]}'
    print(f'save plot to folder: {d}')


'''Usage:
python <this_file>
'''

