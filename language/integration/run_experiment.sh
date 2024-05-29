# Required environment variables:
# TAG: tag for the trail
# TYPE: finetune / prompt / prompt-demo  
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list

# Number of training instances per label
#K=16

# Training steps
MAX_STEP=$max_step

# Validation steps
EVAL_STEP=100

# Task specific parameters
# The default length is 128 and the default number of samples is 16.
# For some tasks, we use longer length or double demo (when using demonstrations, double the maximum length).
# For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
# All those parameters are set arbitrarily by observing the data distributions.
TASK_EXTRA=""


# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take 
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
REAL_BS=32
GS=$(expr $BS / $REAL_BS)


# Use a random number to distinguish different trails (avoid accidental overwriting)
TRIAL_IDTF=$RANDOM
DATA_DIR=../data/from_local/$TASK
modelseed=$modelseed
log_file_store=../log_files/$TASK
model_type=bert
output_dir=../ckpt_paths/$TASK/$TASK-$model_type-$METHOD-$REAL_BS-$LR-seed$modelseed-v1

#uniting_model_ckpt=../ckpt_paths/$TASK/$TASK-$TYPE-$MODELNAME-$METHOD-$REAL_BS-$LR



if [ $MODEL == 'roberta-base' ]; then
    len=128;
elif [ $MODEL == 'gpt2' ]; then
    len=256;
fi    


export CUDA_VISIBLE_DEVICES=0

python uniting_new.py \
  --task_name $TASK \
  --data_dir $DATA_DIR \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict \
  --model_name_or_path /data/yekeming/project/neurips2024/uniting4nlp/models/bert-base-uncased \
  --base_model_path /data/yekeming/project/neurips2024/uniting4nlp/ckpt_paths/$TASK/$TASK-$model_type-finetune-32-1e-3-seed0-v1/finetune \
  --graft_model_path /data/yekeming/project/neurips2024/uniting4nlp/ckpt_paths/$TASK/$TASK-$model_type-finetune-32-1e-3-seed1-v1/finetune \
  --uniting_model_ckpt $output_dir \
  --finetune_model_ckpt $output_dir/finetune \
  --model_type bert \
  --method $METHOD \
  --uniting_epochs 10 \
  --uniting_learning_rate $LR \
  --cache_dir model_files \
  --few_shot_type $TYPE \
  --num_k $K \
  --max_seq_length $len \
  --max_length_per_example $len \
  --per_device_train_batch_size $REAL_BS \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps $GS \
  --learning_rate $LR \
  --max_steps $MAX_STEP \
  --logging_steps $EVAL_STEP \
  --eval_steps $EVAL_STEP \
  --num_train_epochs 10 \
  --output_dir $output_dir \
  --seed $modelseed \
  --tag $TAG \
  --optimizer SGD\
  --use_lm_head $uselmhead\
  --weight_decay 1e-4\
  --log_file_store $log_file_store\
  --use_CLS_linearhead $useCLS \
  --fix_head $fixhead\
  --fix_embeddings $fixembeddings\
  --train_bias_only $train_bias_only\
  --no_train\
  $TASK_EXTRA \
  $1 

# Delete the checkpoint 
# Since we need to run multiple trials, saving all the checkpoints takes 
# a lot of storage space. You can find all evaluation results in `log` file anyway.
#rm -r result/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF-$REAL_BS-$LR \
