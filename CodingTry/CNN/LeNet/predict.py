import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet
from torchvision import transforms

def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),   #将下载图片缩放为32*32
         transforms.ToTensor(), #ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可。
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            #  input[channel] = (input[channel] - mean[channel]) / std[channel]
            #  也就是说(（0,1）-0.5）/0.5=(-1,1)
            
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('LeNet/Lenet.pth'))

    im = Image.open('LeNet/1.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    #print(outputs)
    print(classes[int(predict)])


if __name__ == '__main__':
    main()