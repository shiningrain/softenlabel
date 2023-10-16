for method in {'reg','erm','fmu'}
# method='labelsoft'
do
mode='dp'
# for mode in {'eo','dp'}
# do
dir='/home/zxy/main/DL_fairness/1_test_code/soften_label/adult_exp/result'$mode'-1006-5w-partorigin'
if [ $mode == 'eo' ]
then
value=5
else
value=0.2
fi
echo $mode $method $value $dir
/home/zxy/main/anaconda3/envs/softenlabel/bin/python main.py --mode $mode --method $method --lam $value --save_dir $dir
# done
done