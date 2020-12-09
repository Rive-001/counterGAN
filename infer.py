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

# torch.autograd.set_detect_anomaly(True)
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(device)


##Load Datasets
trans = {}
trans["train"] = transforms.Compose(
    [
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

BATCH_SIZE = 512
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
        data["train"], batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
    ),
    "test": DataLoader(
        data["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
    ),
}
saved = [
    # (
    #     "/home/stars/Code/tarang/idl_proj/counterGAN/Linear_bce_last_counterGAN.pth",
    #     "linear_bce",
    # ),
    # (
    #     "/home/stars/Code/tarang/idl_proj/counterGAN/Linear_MSE_Exp2_last_counterGAN.pth",
    #     "linear_mse",
    # ),
    # (
    #     "/home/stars/Code/tarang/idl_proj/counterGAN/ConvMSE_Linear_last_counterGAN.pth",
    #     "conv_mse",
    # ),
    # (
    #     "/home/stars/Code/tarang/idl_proj/counterGAN/convbce_last_counterGAN.pth",
    #     "conv_bce",
    # ),
    (
        "/home/stars/Code/tarang/idl_proj/counterGAN/linear_bce_randomnoise_exp6.pth",
        "linear_random",
    ),
]

for model_path, exp_name in saved:
    if "conv" in exp_name:
        cgan = counterGAN(device, lr=0.001, linear=False)
    else:
        cgan = counterGAN(device, lr=0.001, linear=True)

    # cgan.load_state_dicts("./Linear_last_counterGAN.pth")
    cgan.load_state_dicts(model_path)

    import accuracy

    print(f"\n{exp_name}")
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

    realdf.to_csv(f"{exp_name}_counter_gan_real_pred.csv")
    advdf.to_csv(f"{exp_name}_counter_gan_adv_pred.csv")

# Linear BCE
# Adv Classifier accuracy: 0.010323660714285714
# Adv Defense Accuracy: 0.14885602678571427
# Real Classifier accuracy: 0.9221833881578947
# Real Defense Accuracy: 0.3101356907894737

# Linear MSE
# Adv Classifier accuracy: 0.010323660714285714
# Adv Defense Accuracy: 0.11872209821428571
# Real Classifier accuracy: 0.9220805921052632
# Real Defense Accuracy: 0.2838199013157895
