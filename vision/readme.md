## Uniting For Vision

#### Pretrain

```bash
bash train.sh
```

Parameter in shell：

```bash
python vision.py \
    --model MobileNetV3 \   # 训练所使用的模型
    --dataset CIFAR100 \    # 训练所使用的数据集
    --epoch 200 \           # 训练轮次数
    --seed 1                # 指定的随机种子
```

Before running, you need to create the folders shown in lines 60 to 65 to store the dataset.

After running, the trained model will be stored in the path shown in line 97.

#### Integration

```bash
python uniting.py
```

Before running the above commands, you need to set the paths in lines 182, 183, 188, and 192 of `uniting.py` to the corresponding values, and set the initialization parameters of the Uniting model in line 195 to the required values.
