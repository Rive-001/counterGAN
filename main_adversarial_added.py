import torch

torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data as data

from model_adversarial_added import *

# from wgan_model_adversarial_added import *

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

experiment_name = "linear_bce_randomnoise_exp6"
USE_WANDB = True
if USE_WANDB:
    import wandb

    os.environ["WANDB_API_KEY"] = open("wandbkey").read()
    wandb.init(project="IDL_PROJECT_COUNTERGAN", name=experiment_name)
    params = wandb.config
    wandb.save("main_adversarial_added.py")
    wandb.save("model_adversarial_added.py")
else:
    params = {}

params["round_name"] = "round0"
params["batch_size"] = 784  # * 6  # 256 * 3
params["lr"] = 1e-3
params["momentum"] = 0.9
params["epochs"] = 100

# torch.autograd.set_detect_anomaly(True)
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(device)


##Load Datasets
trans = {}
trans["train"] = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
trans["test"] = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

Datasets = {}
Datasets["train"] = datasets.CIFAR10(
    root="./", train=True, transform=trans["train"], download=True
)
Datasets["test"] = datasets.CIFAR10(
    root="./", train=False, transform=trans["test"], download=True
)

BATCH_SIZE = params["batch_size"]
dataloaders = {
    x: data.DataLoader(
        Datasets[x], shuffle=True, batch_size=BATCH_SIZE, num_workers=num_workers
    )
    for x in ["train", "test"]
}

# Load Adverserial datasets

traindir = "./baseline_adv2/train"
validdir = "./baseline_adv2/dev"

data = {
    "train": datasets.ImageFolder(root=traindir, transform=trans["train"]),
    "test": datasets.ImageFolder(root=validdir, transform=trans["test"]),
}

# Dataloader iterators, make sure to shuffle
dataloaders_adv = {
    "train": DataLoader(
        data["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    ),
    "test": DataLoader(
        data["test"], batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    ),
}


NUM_EPOCHS = params["epochs"]
if USE_WANDB:
    cgan = counterGAN(device, lr=params["lr"], linear=True, wandb=wandb)
else:
    cgan = counterGAN(device, lr=params["lr"], linear=True)
D_losses, G_losses, T_losses, img_list_adv, img_list_real = cgan.train(
    0, NUM_EPOCHS, dataloaders, dataloaders_adv
)

import accuracy

adv_groundtruth, adv_actual_preds, adv_defense_preds = accuracy.accuracy(
    dataloaders_adv, cgan, BATCH_SIZE
)
real_groundtruth, real_actual_preds, real_defense_preds = accuracy.accuracy(
    dataloaders, cgan, BATCH_SIZE
)

import pandas as pd

realdf = pd.DataFrame(columns=["gt_value", "classifier", "defense_pred"])
realdf.gt_value = real_groundtruth
realdf.classifier = real_actual_preds
realdf.defense_pred = real_defense_preds

advdf = pd.DataFrame(columns=["gt_value", "classifier", "defense_pred"])
advdf.gt_value = adv_groundtruth
advdf.classifier = adv_actual_preds
advdf.defense_pred = adv_defense_preds

realdf.to_csv(f"{experiment_name}_counter_gan_real_pred.csv")
advdf.to_csv(f"{experiment_name}_counter_gan_adv_pred.csv")

# Search learning rate space
# lrs = [10**i for i in np.random.uniform(-5,-2, 20)]
# t_losses = []

# for lr in lrs:

#     cgan = counterGAN(device,lr=lr)
#     D_losses, G_losses, T_losses, img_list_adv, img_list_real = cgan.train(0,20,dataloaders,dataloaders_adv,verbose=False)
#     t_losses.append(T_losses[-1])
#     print('Learning rate:',lr,'Classifer loss:',T_losses[-1])

# best_lr = lrs[t_losses.index(min(t_losses))]
# print('Best learning rate:', best_lr)

