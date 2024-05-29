dataset=$1
model=$2
cuda_num=$3

ITERATION=24300
SNAPSHOT=2430
MAX_EPOCH=20
ARCH_CONF_FILE="configs/${dataset}_conf.json"


GRADIENT_CLIP=5                     # !
BATCH_SIZE=1024

######################################################################
LEARNING_RATE=0.001
CHECKPOINT_PATH=../checkpoint/${dataset}_${model}/base_graft
######################################################################

echo ${CHECKPOINT_PATH}
echo "Model save to ${CHECKPOINT_PATH}"

USER_DEFINED_ARGS="--model=graft_${model} --num_loading_workers=1 --arch_config=${ARCH_CONFIG_CONTENT} --learning_rate=${LEARNING_RATE} \
--max_gradient_norm=${GRADIENT_CLIP} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOIN\
T_PATH} --checkpoint_dir_2=${CHECKPOINT_PATH_2} --arch_config=${ARCH_CONF_FILE} --base_model_path=../checkpoint/NIPS2023/${dataset}_${model}/base/best_auc.pkl \
--graft_model_path=../checkpoint/NIPS2023/${dataset}_${model}/base_seed1/best_auc.pkl"

dataset="../dataset/${dataset}/data"

train_file="${dataset}/train.txt"
test_file="${dataset}/test.txt"
data="${train_file},${test_file}"

export CUDA_VISIBLE_DEVICES=${cuda_num}s
echo ${USER_DEFINED_ARGS}
python ../main/multi_metric_base_graft_train.py \
--dataset=${data} \
${USER_DEFINED_ARGS}

echo "Training done: ${CHECKPOINT_PATH}"

