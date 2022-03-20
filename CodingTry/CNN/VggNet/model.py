import torch
import torch.nn as nn

class VggNet(nn.Module):
    def __init__(self, features, num_classes=5, init_weights=False) -> None:
        super(VggNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  #标准的展平操作，第0维度Bantch不展平
        x = self.classifier(x)
        return x  
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    
def make_features(cfg:list):
    layers = []
    in_channels = 3
    for value in cfg:   #用了value这个参数代表cfgs中键值对的值
        if value == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, value, kernel_size=3, padding=1)]  #会发现VggNet中的卷积设计成了不改变H*W的形式
            layers += [nn.ReLU(True)]
            in_channels = value
    return nn.Sequential(*layers)   #非关键字参数形式传入layers，详见Sequential()的两个实例
    
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
    
model_name = "vgg16"   #超参数，给了四类VggNet网络结构
def vgg(model_name, **kwargs):
    assert model_name in cfgs, "Warning: model name {} not found !".format(model_name)
    cfg = cfgs[model_name]
        
    model = VggNet(make_features(cfg), **kwargs)
    return model