import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import preprocess_adult_data
from model import Net_CENSUS
from utilsnew import *
# from utils import train_eo, evaluate_eo


def run_experiments(method='labelsoft_addreg',mode='dp',lam=0.5, alpha=0.1, beta=None, num_exp=10,save_dir=None,batch=500):
    '''
    Retrain each model for 10 times and report the mean ap and dp.
    '''

    ap = []
    gap = []

    for i in range(num_exp):
        print('On experiment', i)
        # get train/test data
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_adult_data(seed = i)

        # initialize model
        model = Net_CENSUS(input_shape=len(X_train[0])).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        # run experiments
        ap_val_epoch = []
        gap_val_epoch = []
        ap_test_epoch = []
        gap_test_epoch = []
        loss=100
        relax_loss=None
        for j in tqdm(range(10)):# 10 epoch
            # if mode == 'dp':
            if loss<beta and method=='labelsoft_pure' and j%2!=0:
                relax_loss=nn.BCELoss(reduction='none')
                print('using gradient ascent!!')
            else:
                relax_loss=None
            writer = SummaryWriter(f"./tensorboard_Exp/{method}-{lam}-{alpha}")
            loss=train_exp(model, criterion, optimizer, X_train, A_train, y_train, method, mode, lam,batch_size=batch,niter=int(50000/batch),alpha=alpha,beta=beta,baselinelam=lam,relax_loss=relax_loss)
            # print(loss)
            writer.add_scalar("loss", loss, j)
            ap_val, gap_val = evaluate_exp(model, X_val, y_val, A_val,mode)
            ap_test, gap_test = evaluate_exp(model, X_test, y_test, A_test,mode)
            # elif mode == 'eo':
            #     train_eo(model, criterion, optimizer, X_train, A_train, y_train, method, lam,batch_size=batch,niter=int(50000/batch),alpha=0.3)
            #     ap_val, gap_val = evaluate_eo(model, X_val, y_val, A_val)
            #     ap_test, gap_test = evaluate_eo(model, X_test, y_test, A_test)
            # else:
            #     print('not support!!!')
            #     os._exit(0)
            if j > 0:
                ap_val_epoch.append(ap_val)
                ap_test_epoch.append(ap_test)
                gap_val_epoch.append(gap_val)
                gap_test_epoch.append(gap_test)


        # best model based on validation performance
        idx = gap_val_epoch.index(min(gap_val_epoch))
        gap.append(gap_test_epoch[idx])
        ap.append(ap_test_epoch[idx])
        # method=f'ls{batch}'#'lspre'
        # if method!='erm':
        #     log_method=f'ls{batch}'
        # else:
        #     log_method=f'erm'
        log_method=method
        update_results(save_dir,log_method,mode,lam,i,[ap_val_epoch,ap_test_epoch],[gap_val_epoch,gap_test_epoch],ap,gap)
        #TODO: for fix lambda
    # update_final_result(save_dir,i,[ap_val_epoch,ap_test_epoch],[gap_val_epoch,gap_test_epoch])


    print('--------AVG---------')
    print('Average Precision', np.mean(ap))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adult Experiment')
    parser.add_argument('--method', default='labelsoft_pure', type=str, help='labelsoft_addreg/labelsoft_noreg/labelsoft_pure/erm/fmu/reg')
    # fmu: fairmixup, reg: regularized loss, erm: training without process
    parser.add_argument('--mode', default='dp', type=str, help='dp/eo')
    parser.add_argument('--lam', default='0.8', type=str, help='lamda for DP/EO')
    parser.add_argument('--alpha', default='0.1', type=float, help='alpha parameter in soften label')
    parser.add_argument('--num_exp', default=10, type=int, help='repeated experiment num')
    parser.add_argument('--save_dir', default='./resultdp-1012-tmp', type=str, help='save_dir')
    parser.add_argument('--beta', default=0.35, type=float, help='beta value which is used for labelsoften_pure')
    args = parser.parse_args()

    # for bs in [500]:#50,100,200,1000]:
    # # #     for method in ['labelsoft','erm']:
    # #         # run_experiments(method,args.mode,l,save_d`    ir=args.save_dir)

    # if args.method == 'labelsoft':
    #     for l in np.arange(0.2,0.31,0.01):#np.arange(0.1,1.0,0.1):
    #         for s in [1]:#[0.1,0.2,0.3]:#[1]:#[0.1,0.2,0.3]:
    #             tmp_save_dir=args.save_dir+f"-{s}"
    #             # for bs in [50,100,200,500,1000,2000]:
    #             bs=1000
    #             run_experiments(args.method,args.mode,l,s,num_exp=args.num_exp,save_dir=tmp_save_dir,batch=bs)
    # else:
    #     tmp_save_dir=args.save_dir+f"-{args.alpha}"
    #     run_experiments(args.method,args.mode,args.lam,args.alpha,num_exp=args.num_exp,save_dir=tmp_save_dir,batch=500)
    tmp_save_dir=args.save_dir+f"-{args.alpha}"#TODO: for fix lambda
    args.lam=float(args.lam)
    args.alpha=float(args.alpha)
    args.beta=float(args.beta)
    # for l in np.arange(0.05,1.0,0.05):
    #     run_experiments(args.method,args.mode, args.lam ,l,args.beta,num_exp=args.num_exp,save_dir=tmp_save_dir,batch=1000)
    run_experiments(args.method,args.mode, args.lam ,args.alpha,args.beta,num_exp=args.num_exp,save_dir=tmp_save_dir,batch=1000)

    # for l in np.arange(0.2,0.6,0.02):
    #     run_experiments(args.method,args.mode, l ,args.alpha,num_exp=args.num_exp,save_dir=tmp_save_dir,batch=1000)
    # batchsize needs to be divisible by 4
