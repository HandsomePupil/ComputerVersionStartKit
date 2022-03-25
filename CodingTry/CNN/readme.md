涉及到
  >LeNet  (作为第一个网络，还是使用了CPU来进行处理，从AlexNet开始，启用了Data.to.GPU模式)
  >AlexNet  （从此网络开始都采用了花分类数据集做训练和测试，详情见下）
  >VggNet  
  >GoogleNet  
  >ResNet  
  >EfficientNet  
  >ConvNext  
 
等的pytorch复现

***  
### 来自 Wz 老师的分享  
* （1）在data_set文件夹下创建新文件夹"flower_data"
* （2）点击链接下载花分类数据集 [http://download.tensorflow.org/example_images/flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)
* （3）解压数据集到flower_data文件夹下
* （4）执行"split_data.py"脚本自动将数据集划分成训练集train和验证集val    

```
├── flower_data   
       ├── flower_photos（解压的数据集文件夹，3670个样本）  
       ├── train（生成的训练集，3306个样本）  
       └── val（生成的验证集，364个样本） 
```
