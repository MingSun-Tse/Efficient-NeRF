import time, math, os, sys, copy, numpy as np, shutil as sh
from .utils import get_project_path, mkdirs
from collections import OrderedDict
import json, yaml
import logging
import traceback

pjoin = os.path.join


class DoubleWriter():

    def __init__(self, f1, f2):
        self.f1, self.f2 = f1, f2

    def write(self, msg):
        self.f1.write(msg)
        self.f2.write(msg)

    def flush(self):
        self.f1.flush()
        self.f2.flush()


class LogPrinter(object):

    def __init__(self, file, ExpID, print_to_screen=False):
        self.file = file
        self.ExpID = ExpID
        self.print_to_screen = print_to_screen

    def __call__(self, *in_str):
        in_str = [str(x) for x in in_str]
        in_str = " ".join(in_str)
        short_exp_id = self.ExpID[-6:]
        pid = os.getpid()
        current_time = time.strftime("%Y/%m/%d-%H:%M:%S")
        out_str = "[%s %s %s] %s" % (short_exp_id, pid, current_time, in_str)
        print(out_str, file=self.file, flush=True)  # print to txt
        if self.print_to_screen:
            print(out_str)  # print to screen

    def logprint(self, *in_str):  # to keep the interface uniform
        self.__call__(*in_str)

    def accprint(self, *in_str):
        blank = '  ' * int(self.ExpID[-1])
        self.__call__(blank, *in_str)

    def netprint(self, *in_str):  # i.e., print without any prefix
        '''Deprecated. Use netprint in Logger.
        '''
        for x in in_str:
            print(x, file=self.file, flush=True)
            if self.print_to_screen:
                print(x)

    def print(self, *in_str):
        '''print without any prefix
        '''
        for x in in_str:
            print(x, file=self.file, flush=True)
            if self.print_to_screen:
                print(x)

    def print_args(self, args):
        '''
            Example: ('batch_size', 16) ('CodeID', 12defsd2) ('decoder', models/small16x_ae_base/d5_base.pth)
            It will sort the arg keys in alphabeta order, ignoring the upper/lower difference.
        '''
        # build a key map for later sorting
        key_map = {}
        for k in args.__dict__:
            k_lower = k.lower()
            if k_lower in key_map:
                key_map[k_lower + '_' + k_lower] = k
            else:
                key_map[k_lower] = k

        # print in the order of sorted lower keys
        logtmp = ''
        for k_ in sorted(key_map.keys()):
            real_key = key_map[k_]
            logtmp += "('%s': %s) " % (real_key, args.__dict__[real_key])
        self.print(logtmp[:-1] + '\n')  # the last one is blank


class Logger(object):
    '''
        The top logger, which 
            (1) set up all log directories
            (2) maintain the losses and accuracies
    '''

    def __init__(self, args):
        self.args = args

        # set up work folder
        self.ExpID = args.ExpID if hasattr(
            args, 'ExpID') and args.ExpID else self.get_ExpID()
        self.Exps_Dir = 'Experiments'
        if hasattr(self.args, 'Exps_Dir'):
            self.Exps_Dir = self.args.Exps_Dir
        self.set_up_dir()

        self.log_printer = LogPrinter(
            self.logtxt, self.ExpID, self.args.debug
            or self.args.screen_print)  # for all txt logging

        # initial print: save args
        self.print_script()
        self.print_nvidia_smi()
        self.print_git_status()
        self.print_note()
        if (not args.debug) and self.SERVER != '':
            # If self.SERVER != '', it shows this is Huan's computer, then call this func, which is just a small feature to my need.
            # When others use this code, they probably need NOT call this func.
            # self.__send_to_exp_hub() # this function is not very useful. deprecated.
            pass
        args.CodeID = self.get_CodeID()
        self.log_printer.print_args(args)
        self.save_args(args)
        self.cache_model()
        self.n_log_item = 0

    def get_CodeID(self):
        if hasattr(self.args, 'CodeID') and self.args.CodeID:
            return self.args.CodeID
        else:
            f = 'wh_git_status_%s.tmp' % time.time()
            script = 'git status >> %s' % f
            os.system(script)
            x = open(f).readlines()
            x = "".join(x)
            os.remove(f)
            if "Changes not staged for commit" in x:
                self.log_printer(
                    "Warning! Your code is not commited. Cannot be too careful."
                )
                time.sleep(3)

            f = 'wh_CodeID_file_%s.tmp' % time.time()
            script = "git log --pretty=oneline >> %s" % f
            os.system(script)
            x = open(f).readline()
            os.remove(f)
            return x[:8]

    def get_ExpID(self):
        self.SERVER = os.environ["SERVER"] if 'SERVER' in os.environ.keys(
        ) else ''
        TimeID = time.strftime("%Y%m%d-%H%M%S")
        ExpID = 'SERVER' + self.SERVER + '-' + TimeID
        return ExpID

    def set_up_dir(self):
        project_path = pjoin(
            "%s/%s_%s" % (self.Exps_Dir, self.args.project_name, self.ExpID))
        if hasattr(self.args, 'resume_ExpID') and self.args.resume_ExpID:
            project_path = get_project_path(self.args.resume_ExpID)
        if self.args.debug:  # debug has the highest priority. If debug, all the things will be saved in Debug_dir
            project_path = "Debug_Dir"

        self.exp_path = project_path
        self.weights_path = pjoin(project_path, "weights")
        self.gen_img_path = pjoin(project_path, "gen_img")
        self.cache_path = pjoin(project_path, ".caches")
        self.log_path = pjoin(project_path, "log")
        self.logplt_path = pjoin(project_path, "log", "plot")
        self.logtxt_path = pjoin(project_path, "log", "log.txt")
        mkdirs(self.weights_path, self.gen_img_path, self.logplt_path,
               self.cache_path)
        self.logtxt = open(self.logtxt_path, "a+")
        self.script_hist = open(
            '.script_history',
            'a+')  # save local script history, for convenience of check
        sys.stderr = DoubleWriter(
            sys.stderr,
            self.logtxt)  # print to stderr, meanwhile, also print to logtxt

    def print_script(self):
        script = ''
        if hasattr(self.args, 'resume_ExpID') and self.args.resume_ExpID:
            script += '\n\n(********************** Experiment Resumed **********************)\n'
        script += 'cd %s\n' % os.path.abspath(os.getcwd())
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
            script += ' '.join(
                ['CUDA_VISIBLE_DEVICES=%s python' % gpu_id, *sys.argv])
        else:
            script += ' '.join(['python', *sys.argv])
        script += '\n'
        print(script, file=self.logtxt, flush=True)
        print(script, file=sys.stdout, flush=True)
        print(script, file=self.script_hist, flush=True)

    def print_exc(self):
        traceback.print_exc(file=self.logtxt)

    def print_nvidia_smi(self):
        out = pjoin(self.log_path, 'gpu_info.txt')
        script = 'nvidia-smi >> %s' % out
        os.system(script)

    def print_git_status(self):
        out = pjoin(self.log_path, 'git_status.txt')
        script = 'git status >> %s' % out
        try:
            os.system(script)
        except:
            pass

    def print_note(self):
        if hasattr(self.args, 'note') and self.args.note:
            self.ExpNote = f'ExpNote: {self.args.note}'
            print(self.ExpNote, file=self.logtxt, flush=True)
            print(self.ExpNote, file=sys.stdout, flush=True)

    def plot(self, name, out_path):
        self.log_tracker.plot(name, out_path)

    def print(self, step):
        keys, values = self.log_tracker.format()
        k = keys.split("|")[0].strip()
        if k:  # only when there is sth to print, print
            values += " (step = %d)" % step
            if step % (self.args.print_interval * 10) == 0 \
                or len(self.log_tracker.loss.keys()) > self.n_log_item: # when a new loss is added into the loss pool, print
                self.log_printer(keys)
                self.n_log_item = len(self.log_tracker.loss.keys())
            self.log_printer(values)

    def cache_model(self):
        '''
            Save the modle architecture, loss, configs, in case of future check.
        '''
        if self.args.debug: return

        t0 = time.time()
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.log_printer(
            f"==> Caching various config files to '{self.cache_path}'")

        extensions = ['.py', '.json', '.yaml', '.sh', '.txt',
                      '.md']  # files of these types will be cached

        def copy_folder(folder_path):
            for root, dirs, files in os.walk(folder_path):
                if '__pycache__' in root: continue
                for f in files:
                    _, ext = os.path.splitext(f)
                    if ext in extensions:
                        dir_path = pjoin(self.cache_path, root)
                        f_path = pjoin(root, f)
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        if os.path.exists(f_path):
                            sh.copy(f_path, dir_path)

        # copy files in current dir
        [
            sh.copy(f, self.cache_path) for f in os.listdir('.')
            if os.path.isfile(f) and os.path.splitext(f)[1] in extensions
        ]

        # copy dirs in current dir
        ignore = ['__pycache__', 'Experiments', 'Debug_Dir', '.git']
        if hasattr(self.args, 'cache_ignore'):
            ignore += self.args.cache_ignore.split(',')
        [
            copy_folder(d) for d in os.listdir('.')
            if os.path.isdir(d) and d not in ignore
        ]
        self.log_printer(f'==> Caching done (time: {time.time() - t0:.2f}s)')

    def get_project_name(self):
        '''For example, 'Projects/CRD/logger.py', then return CRD
        '''
        file_path = os.path.abspath(__file__)
        return file_path.split('/')[-2]

    def save_args(self, args):
        # with open(pjoin(self.log_path, 'params.json'), 'w') as f:
        #     json.dump(args.__dict__, f, indent=4)
        with open(pjoin(self.log_path, 'params.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f, indent=4)

    def netprint(self, net, comment=''):
        with open(pjoin(self.log_path, 'model_arch.txt'), 'w') as f:
            if comment:
                print('%s:' % comment, file=f)
            print('%s\n' % str(net), file=f, flush=True)
