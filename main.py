import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from model import *
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
cgan.train(0,1,dataloaders)
