for method in {'reg','fmu'}
do
mode='dp'
alpha='0.7'
dir='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/result'$mode'-1012-fixalpha'
for lam in {'0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5','0.55','0.6','0.65','0.7','0.75','0.8','0.85','0.9','0.95'}
do
echo $mode $method $lam $alpha $dir
/home/zxy/main/anaconda3/envs/softenlabel/bin/python main-fixalpha.py --mode $mode --method $method --lam $lam --alpha $alpha --save_dir $dir
done
done