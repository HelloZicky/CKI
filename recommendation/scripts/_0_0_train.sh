#!/bin/bash
date
dataset_list=("amazon_beauty")
echo ${dataset_list}
line_num_list=(7828)
cuda_num_list=(3)
echo ${line_num_list}
length=${#dataset_list[@]}
for ((i=0; i<${length}; i++));
do
{
    dataset=${dataset_list[i]}
    cuda_num=${cuda_num_list[i]}
    for model in din
    do
    {
#
        for type in _0_func_base_train _0_func_duet_train
#          for type in _0_func_mask_train
#          for type in _0_func_base_graft_train
        do
          {
            bash ${type}.sh ${dataset} ${model} ${cuda_num}
          } &
        done
#        } &
#        done
    } &
    done
} &
done
wait # 等待所有任务结束
date