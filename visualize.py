import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils import *
# from utils import train_eo, evaluate_eo
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adult Experiment')
    parser.add_argument('--method', default='labelsoft', type=str, help='labelsoft/erm')
    parser.add_argument('--mode', default='DP', type=str, help='DP/EO')
    parser.add_argument('--lam', default=0.1, type=float, help='Lambda for soften label')
    parser.add_argument('--baseline_dir', default='./resultdp-1006-fixalpha-baseline', type=str, help='save_dir')
    parser.add_argument('--result_dir', default='./resultdp-1012-fixalpha-ascent0.25-0.2', type=str, help='raw results are saved here')
    parser.add_argument('--save_dir', default='./resultdp-1012-fixalpha-ascent0.25-0.2', type=str, help='save_dir')
    parser.add_argument('--start_epoch', default=0, type=int, help='save_dir')
    parser.add_argument('--task', default=1, type=int, help='0: box plot, 1: compare trade-off')
    args = parser.parse_args()

    dir_list=traversalDir_FirstPkl(args.result_dir)
    
    # visualize_results(dir_list,args.method,args.mode,misc='1005-update',size=(30,6),save_dir=args.save_dir)
    if args.task==0:
        visualize_results(dir_list,args.method,args.mode,misc='1005-update',size=(30,6),save_dir=args.save_dir)
    elif args.task==1:
        baseline_dir_list=traversalDir_FirstPkl(args.baseline_dir)
        visualize_lambda_effect(dir_list,baseline_dir_list,args.method,args.mode,misc=f'$\Delta$ {args.mode}',size=(10,6),save_dir=args.save_dir,xlim=(0.0,0.15),ylim=(0.7,0.8),start_epoch=args.start_epoch)
    else:
        print('not support task id')