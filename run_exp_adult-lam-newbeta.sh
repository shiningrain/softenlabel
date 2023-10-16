# for method in {'labelsoft','reg','erm','fmu'}
# do
# method='labelsoft'
# mode='dp'
# for alpha in {'0.8','0.9'}
# do
# dir='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/result'$mode'-1006-fixalpha'
# for lam in {'0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95'}
# do
# echo $mode $method $lam $alpha $dir
# /home/zxy/main/anaconda3/envs/softenlabel/bin/python main-fixalpha.py --mode $mode --method $method --lam $lam --alpha $alpha --save_dir $dir
# done
# done

### Run experiments!!
method='labelsoft_pure'
# 'labelsoft_pure'/'labelsoft_addreg'
mode='dp'
# for alpha in {'0.1','0.15','0.2'}
# do
alpha='0.2'
# WARNING: check lambda / alpha dir name mode for main.py!!!
beta_list=($(seq 0.30 0.01 0.40))
value_list=($(seq 0.05 0.05 1))
for beta in ${beta_list[@]}
do
for lam in ${value_list[@]}
# # {'1.0','1.05','1.1','1.15','1.2','1.25','1.3','1.35','1.4','1.45','1.5','1.55','1.6','1.65','1.7','1.75','1.8','1.85','1.9','1.95'}
do
# bsdir='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/resultdp-1006-fixalpha-baseline'
dir='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/result'$mode'-1016-fixalpha-ascent'
echo $mode $method $lam $alpha $beta $dir 
/home/zxy/main/anaconda3/envs/softenlabel/bin/python main-new.py --mode $mode --method $method --lam $lam --alpha $alpha --beta $beta --save_dir $dir
# /home/zxy/main/anaconda3/envs/softenlabel/bin/python visualize.py  --save_dir $dir --result_dir $dir --baseline_dir $bsdir
# done
done
done


# #### visualize!!!
# method='labelsoft_pure'
# # 'labelsoft_pure'/'labelsoft_addreg'
# mode='dp'
# for alpha in {'0.1','0.15','0.2'}
# do
# # alpha='0.25'
# # WARNING: check lambda / alpha dir name mode for main.py!!!
# beta_list=($(seq 0.25 0.01 0.35))
# for beta in ${beta_list[@]}
# # value_list=($(seq 0.05 0.05 1))
# # for lam in ${value_list[@]}
# # # {'1.0','1.05','1.1','1.15','1.2','1.25','1.3','1.35','1.4','1.45','1.5','1.55','1.6','1.65','1.7','1.75','1.8','1.85','1.9','1.95'}
# do
# bsdir='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/resultdp-1006-fixalpha-baseline'
# dir='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/result'$mode'-1012-fixalpha-ascent'$beta'-'$alpha
# echo $mode $method $lam $alpha $dir
# # /home/zxy/main/anaconda3/envs/softenlabel/bin/python main-new.py --mode $mode --method $method --lam $lam --alpha $alpha --save_dir $dir
# /home/zxy/main/anaconda3/envs/softenlabel/bin/python visualize.py  --save_dir $dir --result_dir $dir --baseline_dir $bsdir
# done
# done
# done
