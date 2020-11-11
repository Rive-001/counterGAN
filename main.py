import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data as data
from model import *
import matplotlib.pyplot as plt
import numpy as np
torch.autograd.set_detect_anomaly(True)
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##Load Datasets
trans = {}
trans['train'] = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trans['test'] = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

Datasets = {}
Datasets['train'] = datasets.CIFAR10(
    root='./',train=True,transform=trans['train'],download=True
)
Datasets['test'] = datasets.CIFAR10(
    root='./',train=False,transform=trans['test'],download=True
)

dataloaders = {x:data.DataLoader(Datasets[x], shuffle=True, batch_size=32, num_workers=num_workers) for x in ['train','test']}


cgan = counterGAN(device)
# cgan.train(0,5,dataloaders)

cgan.load_state_dicts('BestcounterGAN.pth')

x = next(iter(dataloaders['test']))
x_out = cgan.infer(x[0].to(device))

imgs_grid = make_grid(x_out)

fig = plt.figure()
imgs_grid = np.transpose(imgs_grid.cpu().detach().numpy(), (1,2,0))
plt.imsave('Img1.png',imgs_grid)
