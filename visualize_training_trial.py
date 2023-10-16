import os
import pickle
import matplotlib.pyplot as plt
from utils import *

target_dir='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/resultdp-1006-fixalpha-baseline'

pkl_list=traversalDir_FirstPkl(target_dir)
epoch=9
# pkl_list=[pkl for pkl in pkl_list if 'erm' in pkl or 'reg' in pkl or 'fmu' in pkl]

for pkl in pkl_list:
    name=os.path.basename(pkl).split('-')[0]
    if name=='labelsoft':
        tmp_string=os.path.basename(pkl).split('-')[-1][:4]
        name+=f'-{tmp_string}'
    with open(pkl, 'rb') as f:#input,bug type,params
        result_dict = pickle.load(f)


    fig, ax = plt.subplots()

    # Plot loss curves for each training run
    for i in range(10):
        plt.plot(range(epoch), result_dict[i]['ap_epoch_list'][-1], label=f'Exp {i + 1}')

    # Add labels and a legend
    plt.xlabel('Epoch')
    plt.ylabel('Avearge Precision')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.title('AP Curves for 10 Training Runs')
    plt.savefig(os.path.join(target_dir,f'AP-{name}.png'))


    plt.cla()
    fig, ax = plt.subplots()

    # Plot loss curves for each training run
    for i in range(10):
        plt.plot(range(epoch), result_dict[i]['gap_epoch_list'][-1], label=f'Exp {i + 1}')

    # Add labels and a legend
    plt.xlabel('Epoch')
    plt.ylabel('DP Gap')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.title('Gap Curves for 10 Training Runs')
    plt.savefig(os.path.join(target_dir,f'gap-{name}.png'))