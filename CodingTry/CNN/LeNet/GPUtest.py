from torch import device
import torch


# device = torch.device("GPU" if torch.cuda.is_available() else "CPU")
# print(device)

if_cuda = torch.cuda.is_available()
if(if_cuda):
    print(if_cuda)