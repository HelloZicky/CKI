seed=100;
bs=32;
lr=1e-3;
model="roberta-base";
TASK='sst2';
max_step=1000;
method='averaging';
modelseed=0;

for K in 16; do
    TAG=exp \
    TYPE=finetune \
    TASK=$TASK \
    K=$K \
    BS=$bs \
    LR=$lr \
    SEED=$seed \
    modelseed=$modelseed \
    uselmhead=1 \
    useCLS=0 \
    max_step=$max_step \
    fixhead=True \
    fixembeddings=True \
    MODEL=$model \
    train_bias_only=False \
    MODELNAME=$model\
    METHOD=$method \
    bash run_experiment.sh;
done