#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder

# Tool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

#%%
IMAGE_SIZE = 224
BATCH_SIZE = 128

#%%
def Get_Device():
    if(torch.cuda.is_available()):
        print("Device : GPU")
        device = torch.device('cuda:0')
    else:
        print("Device : CPU")
        device = torch.device('cpu')

    return device

#%%
test_transform = transforms.Compose([
                                      transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5))
                                      ])


test_set = ImageFolder(root = r"ml2021spring-hw3\food-11\testing", transform = test_transform)
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE)


#%%
device = Get_Device()

# Load model
model_path = r'model/model.pth'

model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 11)
model.load_state_dict(torch.load(model_path))

model.to(device)


#%%
predictions = []

model.eval()

with torch.no_grad():
    for batch in tqdm(test_loader):
        data, _ = batch

        pred = model(data.to(device))

        # Take the class with greatest logit as prediction and record it.
        predictions.extend(pred.argmax(dim=-1).cpu().numpy().tolist()) # [[0.7, 0.2, 0.1], [0.4, 0.5, 0.2], [0.8, 0.1, 0.1]] => Because dim=-1 => [0, 1, 0]

# Save predictions into the file.
with open("predict.csv", "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")

#%%