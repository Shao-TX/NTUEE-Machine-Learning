#%%
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Data Preprocess
import numpy as np
import pandas as pd

# Some function of training.py
from training import Net, prep_dataloader

#%%
# Testing
def test(test_data, model, device):
    model.eval()                                               # set model to evalutation mode
    preds = []
    for x in test_data:                                  
        x = x.to(device)                                 
        with torch.no_grad():                                  # disable gradient calculation
            pred = model(x)                                    # forward pass (compute output)
            preds.append(pred.detach().cpu().numpy().item())   # collect prediction
    return preds

# 存成 Kaggle 形式
def save_pred(preds, df, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    for i, p in enumerate(preds):
        i+=1 # Pandas 是從 1 開始(但 i 等於 0)
        df.loc[i] = [i-1, p]

    df['id'] = df['id'].astype('Int32') # 因為 csv 是由 String 存入，因此需轉成 Int32 才能上傳 Kaggle
    df.to_csv(file + ".csv",
            index = False)

#%%
if __name__ == "__main__":
    # Use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model Path
    model_path = 't.pth'

    # Data Path
    TEST_PATH  = 'ml2021spring-hw1\covid.test.csv'   # path of testing data

    # Load Dataset
    test_data  = prep_dataloader(TEST_PATH ,mode = 'test' ,batch_size = 1, target_only = False)

    # Load Model
    model = Net(test_data.dataset.dim).to(device)

    # Load Best Model Weight
    ckpt = torch.load(model_path, map_location='cpu')  
    model.load_state_dict(ckpt)

    # CSV Name
    df = pd.DataFrame([], columns = ['id', 'tested_positive'])

    preds = test(test_data, model, device)  # Predict Testing COVID-19 Data

    save_pred(preds, df,'submission')       # Save Prediction file to submission.csv
#%%
print(__name__)
# %%
