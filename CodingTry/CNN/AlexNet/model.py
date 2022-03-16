from turtle import forward
from typing_extensions import Self
import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_classes, init_weights=False):
        super(AlexNet, self).__init__()
        # 和论文原文不同，特征提取的通道数全部减半了～
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2), #input(3,224,224)  output(48,55,55),非整数情况卷积向下取整。
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2,),  #output(48,27,27)
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  #output(128,27,27)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),  #output(128,13,13)
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  #output(192,13,13)
            nn.ReLU(True),
            nn.Conv2d(192, 192,kernel_size=3, padding=1),  #output(192,13,13)
            nn.ReLU(True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  #output(128,13,13)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),  #output(128,6,6)
        )
        self.classifier =nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  #展平操作，衔接卷积和全连接层，Start_dim指从第1维逐步展开，四个维度(Batch,C,H,W)
        x = self.classifier(x)
        return x
    
    # 自定义初始化方法，但其实现有的PYTORCH版本会自动套用这一方法，自己初始化
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    