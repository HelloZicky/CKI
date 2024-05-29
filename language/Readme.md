# KI4NLP

## Introduction of files

File path：/nlp

File structure：


- ckpt_paths: path to save model, run 'mkdir ckpt_paths' before run the script

- data: dataset path

- models: pretrained bert, roberta and so on

- integration: code for training the integrated model

## data path

/nlp/data/from_local/${dataset_name}$

## KI

```shell
cd /nlp
```



1. Finetune the nlp models

   ```shell
   cd integration
   bash run_task.sh
   ```

   Parameters in shell:

   ```shell
   # run_task.sh
   seed=100;
   bs=32;
   lr=1e-3;
   model="roberta-base";
   TASK='sst2';  # choose dataset
   max_step=1000;
   method='finetune';  # which means：finetune/uniting/averaging/ensemble/pruning
   modelseed=0;  # seed=0 & seed=1
   
   # run_experiment.sh
   --model_name_or_path ./models/bert-base-uncased \
   --model_type roberta \  # bert/roberta
   ```

2. integration for nlp models

   ```shell
   cd integration
   bash run_task.sh
   ```

   parameter in shell：

   ```shell
   # run_task.sh
   seed=100;
   bs=32;
   lr=1e-3;
   model="roberta-base";
   TASK='sst2';
   max_step=1000;
   method='uniting';
   modelseed=0;
   
   # run_experiment.sh
   --model_name_or_path ./models/bert-base-uncased \
   --base_model_path ./ckpt_paths/$TASK/$TASK-$model_type-finetune-32-1e-3-seed0-v1/finetune \  # path of finetune model A
   --graft_model_path ./ckpt_paths/$TASK/$TASK-$model_type-finetune-32-1e-3-seed1-v1/finetune \  # path of finetune model B
   --model_type roberta \  # choose the model, bert/roberta
   ```

## Introduction for Model

Model path：./integration/src/models.py

- KIModelForNLP：Please delete the version number when reproducing（*v+num* represents the version of roberta，*v+num+b* represents the version of bert）
- PruningModelForNLP：Model Pruning（*_ro*=roberta）
- EnsembleModelForNLP：Output Ensemble（*_ro*=roberta）