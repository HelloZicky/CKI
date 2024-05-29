# KI4Rec

## Folder Introduction

File structure：

- dataset: dataset folder
- main: main function folder
- scripts: shell folder

## Knowledge Integration

1. Enter the scripts folder

   ```shell
   cd scripts
   ```

2. Train static and dynamic recommendation model

   ```shell
   bash _0_0_train.sh
   
   # type setting = _0_func_base_train _0_func_duet_train
   ```

3. KI between static model and dynamic model

   ```shell
   bash _0_0_train.sh
   
   # type setting = _0_func_mask_train
   ```

4. KI between static model and static model

   ```shell
   bash _0_0_train.sh
   
   # type setting = _0_func_base_graft_train
   ```

## Model Introduction

Model Path：./model

- din: static recommendation model
- meta_din: dynamic recommendation model
- mask-din: static+dynamic integration
- graft_din: static+static integration

