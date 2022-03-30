未使用迁移学习的方式，以
```
LR = 0.0001
Optim = Adam
Epochs = 50
Batch = 64
```
对花分类数据集对ResNet-34进行训练得到模型参数，对随机图片“戴墨镜的向日葵”测试得到如下结果：
```
class: daisy        prob: 0.000389
class: dandelion    prob: 0.00399
class: roses        prob: 0.000457
❤class: sunflowers   prob: 0.995
class: tulips       prob: 0.000152
```

2080Ti ,num_worker=8 ,大约用时13分钟,MAX(val_accuracy) = 0.831
