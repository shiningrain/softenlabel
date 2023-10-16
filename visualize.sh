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

mode='dp'
value_list=($(seq 0.3 0.1 0.7))
for lam in ${value_list[@]}
# {'1.0','1.05','1.1','1.15','1.2','1.25','1.3','1.35','1.4','1.45','1.5','1.55','1.6','1.65','1.7','1.75','1.8','1.85','1.9','1.95'}
do
dir='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/result'$mode'-1006-fixalpha-'$lam
for start in {0,1}
do
echo $mode $lam $dir $start
/home/zxy/main/anaconda3/envs/softenlabel/bin/python visualize.py --mode $mode --lam $lam --save_dir $dir --result_dir $dir --start_epoch $start
done
done
