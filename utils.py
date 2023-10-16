import torch
import numpy as np
from numpy.random import beta
from sklearn.metrics import average_precision_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import os
import pickle
import matplotlib.pyplot as plt
import copy
from fairlearn.metrics import MetricFrame, demographic_parity_difference,equalized_odds_difference
from tqdm import trange
import heapq

def sample_batch_sen_idx(X, A, y, batch_size, s):    
    batch_idx = np.random.choice(np.where(A==s)[0], size=batch_size, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()
    if s==0:
        attribute = list(['a' for i in range(batch_size)])
    elif s==1:
        attribute = list(['b' for i in range(batch_size)])
    return batch_x, batch_y,attribute

def sample_batch_idx(X, A, y, batch_size):    
    batch_idx = np.random.choice(np.where(A!=None)[0], size=batch_size, replace=False).tolist()
    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    attribute = list(A[batch_idx])
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()
    return batch_x, batch_y,attribute

def sample_batched(X_train, A_train, y_train, batch_size,niter):
    batch_x_list=[]
    batch_y_list=[]
    batch_attribute_list=[]
    for i in range(niter):
        if i==niter-1:
            batch_x_list.append(torch.tensor(X_train[batch_size*i:]).cuda().float())
            batch_y_list.append(torch.tensor(y_train[batch_size*i:]).cuda().float())
            batch_attribute_list.append(A_train[batch_size*i:])
        else:
            batch_x_list.append(torch.tensor(X_train[batch_size*i:batch_size*(i+1)]).cuda().float())
            batch_y_list.append(torch.tensor(y_train[batch_size*i:batch_size*(i+1)]).cuda().float())
            batch_attribute_list.append(A_train[batch_size*i:batch_size*(i+1)])
    return batch_x_list,batch_y_list,batch_attribute_list

def sample_batch_sen_idx_y(X, A, y, batch_size, s):
    batch_idx = []
    attribute = []
    for i in range(2):
        idx = list(set(np.where(A==s)[0]) & set(np.where(y==i)[0]))
        batch_idx += np.random.choice(idx, size=batch_size, replace=False).tolist()
        if i == 0:
            attribute += list(['a' for i in range(batch_size)])
        if i == 1:
            attribute += list(['b' for i in range(batch_size)])

    batch_x = X[batch_idx]
    batch_y = y[batch_idx]
    batch_x = torch.tensor(batch_x).cuda().float()
    batch_y = torch.tensor(batch_y).cuda().float()

    return batch_x, batch_y, attribute

def compute_delta_fl(model,batch_x,batch_y,attribute,mode):
    pred_y = model(batch_x).detach().cpu().numpy()
    # y_true=batch_y.cpu().numpy()
    binary_y_pred=binary_value(pred_y)
    binary_y_true=binary_value(batch_y)# turn to binary or delta wil be 0
    if mode=='dp':
        delta=demographic_parity_difference(y_true=binary_y_true,y_pred=binary_y_pred,sensitive_features=attribute)
    elif mode=='eo':
        delta=equalized_odds_difference(y_true=binary_y_true,y_pred=binary_y_pred,sensitive_features=attribute)
    # print(delta)
    return delta,pred_y

def compute_dp(model,batch_x_0,batch_x_1):
    output_0 = model(batch_x_0)
    output_1 = model(batch_x_1)
    delta_dp=torch.abs(output_0.mean() - output_1.mean())
    return delta_dp

def binary_value(y_list,threshold=0.5):
    output=[]
    for y in y_list:
        if y >0.5:
            output.append(1)
        else:
            output.append(0)
    return output

# def compute_delta_batch(model,batch_x_0,batch_x_1,attribute,mode,pred_y=None):
#     if mode =='dp': 
#         pred_y_raw_0 = model(batch_x_0)
#         pred_y_raw_1 = model(batch_x_1)
#         delta=torch.abs()

#     return delta,pred_y_raw
    

def soften_batch_y(batch_y,y_pre,alpha):
    return (1-alpha)*batch_y+y_pre*alpha # knowledge distilltion from previous model

def regloss(model, batch_x_0, batch_x_1,mode,batch_x=None):
    if mode=='dp':   
        output_0 = model(batch_x_0)
        output_1 = model(batch_x_1)
        loss_reg = output_0.mean() - output_1.mean()
        # y_pre=torch.cat((output_0, output_1), 0)
        if batch_x==None:
            y_pre=None
        else:
            y_pre=model(batch_x)
    elif mode == 'eo':
        loss_reg = 0
        for i in range(2):
            batch_x_0_i = batch_x_0[i]
            batch_x_1_i = batch_x_1[i]

            output_0 = model(batch_x_0_i)
            output_1 = model(batch_x_1_i)
            loss_reg += torch.abs(output_0.mean() - output_1.mean())
        if batch_x==None:
            y_pre=None
        else:
            y_pre=model(batch_x)
    return loss_reg,y_pre

def fairmixup(model, batch_x_0, batch_x_1,mode):
    loss_reg = 0
    alpha = 1
    if mode=='dp':
        gamma = beta(alpha, alpha)

        batch_x_mix = batch_x_0 * gamma + batch_x_1 * (1 - gamma)
        batch_x_mix = batch_x_mix.requires_grad_(True)
        output = model(batch_x_mix)
        # gradient regularization
        gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]

        batch_x_d = batch_x_1 - batch_x_0
        grad_inn = (gradx * batch_x_d).sum(1)
        E_grad = grad_inn.mean(0)
        loss_reg = torch.abs(E_grad)
    elif mode == 'eo':
        for i in range(2):
            gamma = beta(alpha, alpha)
            batch_x_0_i = batch_x_0[i]
            batch_x_1_i = batch_x_1[i]
            batch_x_mix = batch_x_0_i * gamma + batch_x_1_i * (1 - gamma)
            batch_x_mix = batch_x_mix.requires_grad_(True)
            output = model(batch_x_mix)
            # gradient regularization
            gradx = torch.autograd.grad(output.sum(), batch_x_mix, create_graph=True)[0]
            batch_x_d = batch_x_1_i - batch_x_0_i
            grad_inn = (gradx * batch_x_d).sum(1)
            loss_reg += torch.abs(grad_inn.mean())
    else:
        print('fairmixup does not support this method')
    return loss_reg

def add_reg_y(lam,delta,batch_y_soft,sensitive_attribute):
    tmp_index=int(len(sensitive_attribute)/2)
    batch_y_0=batch_y_soft[:tmp_index]
    batch_y_1=batch_y_soft[tmp_index:]
    # if batch_y_0.mean()>batch_y_1.mean():
        # new_batch_y_0=batch_y_0-lam*delta/2
        # new_batch_y_1=batch_y_1+lam*delta/2
    # else:
    #     new_batch_y_0=batch_y_0+lam*delta/2
    #     new_batch_y_1=batch_y_1-lam*delta/2
    new_batch_y_0=batch_y_0-lam*delta/2 # if batch_y_0 mean >batch_y_1 mean, delta is positive, else negative
    new_batch_y_1=batch_y_1+lam*delta/2
    
    new_batch_y=torch.cat((new_batch_y_0, new_batch_y_1), 0)
    # zero_metric=torch.zeros_like(new_batch_y)
    # one_metric=torch.ones_like(new_batch_y)
    # new_batch_y=torch.where(new_batch_y > 1, one_metric, new_batch_y)
    # new_batch_y=torch.where(new_batch_y < 0, zero_metric, new_batch_y) # clip xx<0 and xx>1
    new_batch_y=torch.clamp(new_batch_y, min=0.0, max=1.0)
    return new_batch_y


def train_exp(model, criterion, optimizer, X_train, A_train, y_train, method, mode, lam, batch_size=500, niter=100,alpha=0.1,baselinelam=0.5):
    model.train()
    model_pre=copy.deepcopy(model)
    # batched_x, batched_y, batched_sensitive_attribute = sample_batched(X_train, A_train, y_train, batch_size,niter)
    
    for it in range(niter):
        loss_reg=0
        if mode == 'dp': # TODO: back
            tmp_batch=int(batch_size/2)
            batch_x_0, batch_y_0, sensitive_attribute_0 = sample_batch_sen_idx(X_train, A_train, y_train, tmp_batch, 0)
            batch_x_1, batch_y_1, sensitive_attribute_1 = sample_batch_sen_idx(X_train, A_train, y_train, tmp_batch, 1)
            batch_x = torch.cat((batch_x_0, batch_x_1), 0)
            batch_y = torch.cat((batch_y_0, batch_y_1), 0)
            sensitive_attribute = sensitive_attribute_0+sensitive_attribute_1
        if mode == 'eo':
            # Gender Split
            tmp_batch=int(batch_size/4)
            batch_x_0, batch_y_0,sensitive_attribute_0 = sample_batch_sen_idx_y(X_train, A_train, y_train, tmp_batch, 0)
            batch_x_1, batch_y_1,sensitive_attribute_1 = sample_batch_sen_idx_y(X_train, A_train, y_train, tmp_batch, 1)
            batch_x_0 = [batch_x_0[:tmp_batch], batch_x_0[tmp_batch:]]
            sensitive_attribute_0 = [sensitive_attribute_0[:tmp_batch], sensitive_attribute_0[tmp_batch:]]
            batch_x_1 = [batch_x_1[:tmp_batch], batch_x_1[tmp_batch:]]
            sensitive_attribute_1 = [sensitive_attribute_1[:tmp_batch], sensitive_attribute_1[tmp_batch:]]
            batch_x = torch.cat((batch_x_0[0],batch_x_0[1], batch_x_1[0], batch_x_1[1]), 0)
            batch_y = torch.cat((batch_y_0, batch_y_1), 0)
            sensitive_attribute=sensitive_attribute_0[0]+sensitive_attribute_0[1]+sensitive_attribute_1[0]+sensitive_attribute_1[1]
        # tmp_batch=int(batch_size/2)
        # batch_x_0, batch_y_0, sensitive_attribute_0 = sample_batch_sen_idx(X_train, A_train, y_train, tmp_batch, 0)
        # batch_x_1, batch_y_1, sensitive_attribute_1 = sample_batch_sen_idx(X_train, A_train, y_train, tmp_batch, 1)
        # batch_x = torch.cat((batch_x_0, batch_x_1), 0)
        # batch_y = torch.cat((batch_y_0, batch_y_1), 0)
        # sensitive_attribute = sensitive_attribute_0+sensitive_attribute_1

        ## softlabel before        
        if method == 'labelsoft':
            _,y_pre=regloss(model_pre, batch_x_0, batch_x_1,mode=mode,batch_x=batch_x)
            model_pre=copy.deepcopy(model) # update, use previous model to calculate DP/EO
            batch_y_soft=soften_batch_y(batch_y,y_pre,alpha)
            loss_reg,_=regloss(model, batch_x_0, batch_x_1,mode=mode)
            batch_y=add_reg_y(lam,loss_reg,batch_y_soft,sensitive_attribute)
            # batch_y=soften_batch_y(batch_y,delta,alpha)
        elif method == 'erm':
            pass
        elif method == 'reg':
            # loss_reg,_ = compute_delta_fl(model, batch_x, batch_y,attribute=sensitive_attribute,mode=mode)
            # print(loss_reg)
            loss_reg,_=regloss(model, batch_x_0, batch_x_1,mode=mode)
            loss_reg=torch.abs(loss_reg)
        elif method == 'fmu':
            loss_reg = fairmixup(model, batch_x_0, batch_x_1,mode=mode)
        else:
            print('not support !!!')
            os._exit(0)

        output = model(batch_x)
        loss_sup = criterion(output, batch_y)

        # ## softlabel after      
        # if method == 'labelsoft':
        #     if it<niter-1:
        #         # if mode=='dp':
        #         delta=compute_delta_fl(None, batch_x, batch_y,attribute=sensitive_attribute,mode=mode,pred_y=output)
        #         # elif mode=='eo':
        #         #     delta=compute_delta_fl(model, batch_x, batch_y,attribute=sensitive_attribute)
        #         tmp_lam=lam*alpha
        #         batched_y[it+1]=soften_batch_y(batched_y[it+1],delta,tmp_lam)
        # elif method == 'erm':
        #     pass
        # else:
        #     print('not support !!!')
        #     os._exit(0)

        # final loss
        if method == 'fmu' or method == 'reg':
            loss = loss_sup + baselinelam*loss_reg
            # print(loss_reg)
        else:
            loss = loss_sup

        model_pre=copy.deepcopy(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss



def evaluate_exp(model, X_test, y_test, A_test,mode):
    model.eval()

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    Y_test_cuda = torch.tensor(y_test).cuda().float()
    output = model(X_test_cuda)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    attribute=[]
    for a in A_test:
        if a == 0.0:
            attribute.append('a')
        elif a==1.0:
            attribute.append('b')
        else:
            print('A_test out of boundary!!!')
    gap,_=compute_delta_fl(model,X_test_cuda,Y_test_cuda,attribute,mode=mode)
    return ap, gap


def update_results(save_dir,method,mode,lam,i,ap_epoch_list,gap_epoch_list,ap,gap):
    # ap_epoch_list=[ap val, ap test]
    # gap_epoch_list=[gap val, gap test]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path=os.path.join(save_dir,f'{method}-{mode}-{lam}.pkl')
    if not os.path.exists(save_path):
        save_dict={}
    else:
        with open(save_path, 'rb') as f:#input,bug type,params
            save_dict = pickle.load(f)

    save_dict[i]={}
    save_dict[i]['ap_epoch_list']=ap_epoch_list
    save_dict[i]['gap_epoch_list']=gap_epoch_list
    save_dict['ap']=ap
    save_dict['gap']=gap
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)


def visualize_results(result_path_list,method,mode,misc='',size=None,save_dir=None):
    ap_list=[]
    gap_list=[]
    label_list=[]
    result_path_list.sort()
    result_path_list.insert(2,result_path_list[-1])
    result_path_list=result_path_list[:-1]
    for path in result_path_list:
        base_name=os.path.basename(path)
        tmp_list=base_name.split('-')
        method,mode,lam=tmp_list[0],tmp_list[1],tmp_list[-1]
        lam=round(float(lam.replace('.pkl','')),2)
        if 'labelsoft' in method:
            method='ls'
        with open(path, 'rb') as f:#input,bug type,params
            result_dict = pickle.load(f)
        ap_list.append(result_dict['ap'])
        gap_list.append([float(gap) for gap in result_dict['gap']])
        if method=='erm':
            label_list.append(f"{method}")
        else:
            label_list.append(f"{method}-{lam}")

    # generate delta dp boxplot
    if size!=None:
        plt.figure(figsize=size)
    elif len(label_list)>12:
        plt.figure(figsize=(20,4))
    else:
        plt.figure(figsize=(10,4))
    plt.grid(True)  # 显示网格
    plt.boxplot(gap_list,
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=label_list)
    if save_dir==None:
        plt.savefig(f'./deltaDP-{mode}{misc}.png',dpi=300)
    else:
        plt.savefig(os.path.join(save_dir,f'deltaDP-{mode}{misc}.png'),dpi=300)

    # generate ap boxplot
    plt.cla()
    if size!=None:
        plt.figure(figsize=size)
    elif len(label_list)>12:
        plt.figure(figsize=(20,4))
    else:
        plt.figure(figsize=(10,4))
    plt.grid(True)  # 显示网格
    plt.boxplot(ap_list,
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=label_list)
    if save_dir==None:
        plt.savefig(f'./ap-{mode}{misc}.png',dpi=300)
    else:
        plt.savefig(os.path.join(save_dir,f'ap-{mode}{misc}'),dpi=300)


def visualize_lambda_effect(result_path_list,baseline_path_list,method,mode,misc='',size=None,save_dir=None,ylim=None,xlim=None,start_epoch=None):
    color_list=['red','blue','green','orange','purple','yellow','brown']
    result_dict_path=os.path.join(save_dir,'result.pkl')
    if start_epoch==0:
        start_epoch=None
    # if os.path.exists(result_dict_path):
    #     with open(result_dict_path, 'rb') as f:#input,bug type,params
    #         result_dict = pickle.load(f)
    # else:
    result_dict={}
    result_path_list=[i for i in result_path_list if 'result.pkl' not in i]
    baseline_result_path_list=[i for i in baseline_path_list]
    erm=False
    for path in baseline_result_path_list:
        base_name=os.path.basename(path)
        tmp_list=base_name.split('-')
        method,mode,lam=tmp_list[0],tmp_list[1],tmp_list[-1]
        if method not in result_dict.keys():
            result_dict[method]=[]
        if method=='erm':
            if erm:
                continue
            else:
                erm=True
        lam=round(float(lam.replace('.pkl','')),2)
        with open(path, 'rb') as f:#input,bug type,params
            tmp_dict = pickle.load(f)
        if start_epoch==None:
            result_dict[method].append((np.mean(tmp_dict['ap']),np.mean([float(gap) for gap in tmp_dict['gap']]),lam))
        else:
            result_tuple=get_later_result(start_epoch,tmp_dict,lam)
            result_dict[method].append(result_tuple)

    for path in result_path_list:
        base_name=os.path.basename(path)
        tmp_list=base_name.split('-')
        method,mode,lam=tmp_list[0],tmp_list[1],tmp_list[-1]
        if method not in result_dict.keys():
            result_dict[method]=[]
        lam=round(float(lam.replace('.pkl','')),2)
        with open(path, 'rb') as f:#input,bug type,params
            tmp_dict = pickle.load(f)
        if start_epoch==None:
            result_dict[method].append((np.mean(tmp_dict['ap']),np.mean([float(gap) for gap in tmp_dict['gap']]),lam))
        else:
            result_tuple=get_later_result(start_epoch,tmp_dict,lam)
            result_dict[method].append(result_tuple)

    with open(result_dict_path, 'wb') as f:
        pickle.dump(result_dict, f)
    plt.figure(figsize=(10,10))
    plt.grid(True)  # 显示网格
    tmp_dict={}
    key_list=list(result_dict.keys())
    for k in range(len(key_list)):
        key=key_list[k]
        tmp_dict[key]=[(i[0],i[1]) for i in result_dict[key]]
        tmp_dict[key]=sorted(tmp_dict[key],key=lambda x:x[0])
        ap,gap=zip(*tmp_dict[key])
        plt.plot(gap, ap, label=key, marker='o', color=color_list[k], linestyle='-', linewidth=3)
    plt.xlabel(misc)
    plt.ylabel('AP')
    plt.legend()
    title='-'.join(os.path.basename(save_dir).split('-')[2:])
    if 'ascent0' in title:
        value=title.split('ascent')[-1].split('-')
        title=f'threshold={value[0]},alpha={value[1]}'
    else:
        value=title.split('-')[-1]
        title=f'alpha={value}'
    plt.title(f'{title}')
    if xlim!=None:
        plt.xlim(xlim[0],xlim[1])
    if ylim!=None:
        plt.ylim(ylim[0],ylim[1])
    if save_dir==None:
        plt.savefig(f'./delta-{mode}.png',dpi=300)
    elif start_epoch==None:
        plt.savefig(os.path.join(save_dir,f'delta-{mode}.png'),dpi=300)
    else:
        plt.savefig(os.path.join(save_dir,f'delta-{mode}-{start_epoch}.png'),dpi=300)

def get_later_result(start_epoch,tmp_dict,lam):
    tmp_ap=[]
    tmp_gap=[]
    for key in tmp_dict.keys():
        if isinstance(key,int):
            gap_val_epoch=tmp_dict[key]['gap_epoch_list'][0][start_epoch:]
            ap_test_epoch=tmp_dict[key]['ap_epoch_list'][1][start_epoch:]
            gap_test_epoch=tmp_dict[key]['gap_epoch_list'][1][start_epoch:]
            idx = gap_val_epoch.index(min(gap_val_epoch))
            tmp_gap.append(gap_test_epoch[idx])
            tmp_ap.append(ap_test_epoch[idx])
    tmp_result=(np.mean(tmp_ap),np.mean(tmp_gap),lam)
    return tmp_result

def traversalDir_FirstPkl(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if (os.path.isfile(m) and '.pkl' in m):
                tmplist.append(m)
    return tmplist

def traversalDir_FirstDir(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if os.path.isdir(m):
                tmplist.append(m)
    return tmplist

def filter_results(ap_list,fair_list,top=10,positive_ap=True):

    if positive_ap:
        ap_max_num_index=[i for i in range(len(ap_list)) if ap_list[i]>0]
    else:
        ap_max_number = heapq.nlargest(top, ap_list)
        ap_max_num_index = list(map(ap_list.index, ap_max_number))
    fair_max_number = heapq.nlargest(top, fair_list)
    fair_max_num_index = list(map(fair_list.index, fair_max_number))
    best_results = list(set(ap_max_num_index).intersection(set(fair_max_num_index)))
    if best_results==[]:
        return fair_max_num_index
    return best_results

def get_best_results(dir_list,baseline_pkl,misc='',size=None,save_dir=None,specific_pkl=[]):
    pkl_list=[]
    with open(baseline_pkl, 'rb') as f:#input,bug type,params
        baseline_results = pickle.load(f)
    baseline_ap=np.mean(baseline_results['ap'])
    baseline_fairness=np.mean(baseline_results['gap'])
    for root_dir in dir_list:
        # dirname=traversalDir_FirstDir(root_dir)
        pkl_list+=traversalDir_FirstPkl(root_dir)

    total_ap_list=[]
    ap_list=[]
    total_fairness_list=[]
    fair_list=[]
    label_list=[]
    specific_index=[]

    for p in trange(len(pkl_list)):
        path=pkl_list[p]
        
        base_name=os.path.basename(path)
        tmp_list=base_name.split('-')
        method,mode,lam=tmp_list[0],tmp_list[1],tmp_list[-1]
        scale=os.path.dirname(path).split('-')[-1]
        lam=round(float(lam.replace('.pkl','')),2)
        if 'labelsoft' in method:
            method='ls'
        with open(path, 'rb') as f:#input,bug type,params
            result_dict = pickle.load(f)
        ap_list.append(np.mean(result_dict['ap'])-baseline_ap) # acc improvment 
        total_ap_list.append(result_dict['ap'])
        fair_list.append(baseline_fairness-np.mean([float(gap) for gap in result_dict['gap']]))# bias reduction
        total_fairness_list.append([float(gap) for gap in result_dict['gap']])
        if method=='erm':
            label_list.append(f"{method}")
        else:
            label_list.append(f"{method}-{lam}-{scale}")

        if path in specific_pkl:
            specific_index.append(len(label_list)-1)

    index_list=filter_results(ap_list,fair_list,top=10,positive_ap=False)
    final_fairness_list=[total_fairness_list[i] for i in index_list]
    final_fairness_list+=[total_fairness_list[i] for i in specific_index if i not in index_list]
    final_fairness_list.append(baseline_results['gap'])
    final_ap_list=[total_ap_list[i] for i in index_list]
    final_ap_list+=[total_ap_list[i] for i in specific_index if i not in index_list]
    final_ap_list.append(baseline_results['ap'])
    final_label_list=[label_list[i] for i in index_list]
    final_label_list+=[label_list[i] for i in specific_index if i not in index_list]
    final_label_list.append('erm')

    # generate delta dp boxplot
    if size!=None:
        plt.figure(figsize=size)
    elif len(label_list)>12:
        plt.figure(figsize=(20,4))
    else:
        plt.figure(figsize=(10,4))
    plt.grid(True)  # 显示网格
    plt.boxplot(final_fairness_list,
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=final_label_list)
    if save_dir==None:
        plt.savefig(f'./deltaDP-{mode}{misc}.png',dpi=300)
    else:
        plt.savefig(os.path.join(save_dir,f'deltaDP-{mode}{misc}.png'),dpi=300)

    # generate ap boxplot
    plt.cla()
    if size!=None:
        plt.figure(figsize=size)
    elif len(label_list)>12:
        plt.figure(figsize=(20,4))
    else:
        plt.figure(figsize=(10,4))
    plt.grid(True)  # 显示网格
    plt.boxplot(final_ap_list,
                medianprops={'color': 'red', 'linewidth': '1.5'},
                meanline=True,
                showmeans=True,
                meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                labels=final_label_list)
    if save_dir==None:
        plt.savefig(f'./ap-{mode}{misc}.png',dpi=300)
    else:
        plt.savefig(os.path.join(save_dir,f'ap-{mode}{misc}'),dpi=300)
