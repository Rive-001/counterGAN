import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data as data
from model_adversarial_added import *
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
torch.autograd.set_detect_anomaly(True)
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
##Load Datasets
trans = {}
trans['train'] = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
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

BATCH_SIZE = 784
dataloaders = {x:data.DataLoader(Datasets[x], shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers) for x in ['train','test']}

#Load Adverserial datasets

traindir = "./baseline_adv2/train"
validdir = "./baseline_adv2/dev"

data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=trans['train']),
    'test':
    datasets.ImageFolder(root=validdir, transform=trans['test'])
}

# Dataloader iterators, make sure to shuffle
dataloaders_adv = {
    'train': DataLoader(data['train'], batch_size=BATCH_SIZE, shuffle=True,num_workers=num_workers),
    'test': DataLoader(data['test'], batch_size=BATCH_SIZE, shuffle=True,num_workers=num_workers)
}
    






cgan = counterGAN(device)
D_losses, G_losses, img_list_adv, img_list_real = cgan.train(0,100,dataloaders,dataloaders_adv)
#D_losses,G_losses,img_list = cgan.train(0,50,dataloaders)
# cgan.visualize_images(img_list_adv, )

# cgan.load_state_dicts('BestcounterGAN.pth')

# x = next(iter(dataloaders['test']))
# x_out = cgan.infer(x[0].to(device))

# imgs_grid = make_grid(x_out)

# fig = plt.figure()
# imgs_grid = np.transpose(imgs_grid.cpu().detach().numpy(), (1,2,0))
# plt.imsave('Img1.png',imgs_grid)
