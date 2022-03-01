#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder

# Tool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from torchvision.utils import save_image

#%%
IMAGE_SIZE = 224
BATCH_SIZE = 1


#%%
train_transform = transforms.Compose([
                                      transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                      transforms.ColorJitter(brightness=(0.5, 1.2)),            # 隨機亮度調整
                                      transforms.RandomHorizontalFlip(p=0.5),                   # 隨機水平翻轉
                                      transforms.RandomRotation((-40, 40)),                            # 隨機旋轉
                                      transforms.RandomResizedCrop(size = 224, scale = (0.5, 1,5)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5))
                                      ])

train_set = ImageFolder(root = r"ml2021spring-hw3\food-11\training\labeled", transform = train_transform)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)

i = 1

#%%
for data, label in tqdm(train_loader):
    i += 1
    save_image(data, "ArgIMG/img_" + str(i) + ".jpg", normalize=True)

#%%