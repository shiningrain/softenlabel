import os
import pickle

path1='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/resultdp-1010-fixalpha-addreg1-0.15/labelsoft_addreg-dp-0.05.pkl'
path2='/home/zxy/main/DL_fairness/1_test_code/soften_label/resulteo-0914-1/ls1000-eo-0.5.pkl'

with open(path1, 'rb') as f:#input,bug type,params
    save_dict1 = pickle.load(f)
with open(path2, 'rb') as f:#input,bug type,params
    save_dict2 = pickle.load(f)

print(1)
os._exit(0)