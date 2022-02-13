#%%
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# tqdm
from tqdm import tqdm

# Data Preprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#%%
class TIMITDataset(Dataset):
    def __init__(self, x, y, test_mode=False):
        self.test_mode = test_mode

        if(self.test_mode == False):
            self.data  = torch.FloatTensor(x)
            self.label = torch.LongTensor(y) # Beacuse CrossEntropy need Long type
        else:
            self.data  = torch.FloatTensor(x)
    
    def __getitem__(self, index):
        if(self.test_mode == False):
            return self.data[index], self.label[index]
        else:
            return self.data[index]
    
    def __len__(self):
        return len(self.data)

#%%
class TIMITDataset(Dataset):
    def __init__(self, x, y, test_mode=False):
        self.test_mode = test_mode

        if(self.test_mode == False):
            self.data  = torch.FloatTensor(x)
            self.label = torch.LongTensor(y) # Beacuse CrossEntropy need Long type
        else:
            self.data  = torch.FloatTensor(x)
    
    def __getitem__(self, index):
        if(self.test_mode == False):
            return self.data[index], self.label[index]
        else:
            return self.data[index]
    
    def __len__(self):
        return len(self.data)

#%%
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.layer1   = nn.Linear(429 , 1024)
        self.layer2   = nn.Linear(1024, 512)
        self.layer3   = nn.Linear(512 , 256)
        self.layer4   = nn.Linear(256 , 128)
        self.layer5   = nn.Linear(128 , 64)
        self.out      = nn.Linear(64 , 39)

        self.act_fn_1 = nn.Sigmoid()
        self.act_fn_2 = nn.ReLU()

        self.BN1      = nn.BatchNorm1d(1024)
        self.BN2      = nn.BatchNorm1d(512)
        self.BN3      = nn.BatchNorm1d(256)
        self.BN4      = nn.BatchNorm1d(128)
        self.BN5      = nn.BatchNorm1d(64)

        self.dropout  = nn.Dropout(p = 0.5) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.BN1(x)
        x = self.dropout(x)
        x = self.act_fn_2(x)

        x = self.layer2(x)
        x = self.BN2(x)
        x = self.dropout(x)
        x = self.act_fn_2(x)

        x = self.layer3(x)
        x = self.BN3(x)
        x = self.act_fn_2(x)

        x = self.layer4(x)
        x = self.BN4(x)
        x = self.act_fn_2(x)

        x = self.layer5(x)
        x = self.BN5(x)
        x = self.act_fn_2(x)

        x = self.out(x)

        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.layer1   = nn.Linear(429 , 1024)
        self.layer2   = nn.Linear(1024, 512)
        self.layer3   = nn.Linear(512 , 256)
        self.layer4   = nn.Linear(256 , 128)
        self.out      = nn.Linear(128 , 39)

        self.act_fn_1 = nn.Sigmoid()
        self.act_fn_2 = nn.ReLU()

        self.BN1      = nn.BatchNorm1d(1024)
        self.BN2      = nn.BatchNorm1d(512)
        self.BN3      = nn.BatchNorm1d(256)
        self.BN4      = nn.BatchNorm1d(128)

        self.dropout  = nn.Dropout(p = 0.5) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.BN1(x)
        x = self.dropout(x)
        x = self.act_fn_2(x)

        x = self.layer2(x)
        x = self.BN2(x)
        x = self.dropout(x)
        x = self.act_fn_2(x)

        x = self.layer3(x)
        x = self.BN3(x)
        x = self.act_fn_2(x)

        x = self.layer4(x)
        x = self.BN4(x)
        x = self.act_fn_2(x)

        x = self.out(x)

        return x

#%%
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.layer1   = nn.Linear(429 , 2048)
        self.layer2   = nn.Linear(2048 , 1024)
        self.layer3   = nn.Linear(1024, 512)
        self.layer4   = nn.Linear(512 , 256)
        self.layer5   = nn.Linear(256 , 128)
        self.layer6   = nn.Linear(128 , 64)
        self.out      = nn.Linear(64 , 39)

        self.act_fn_1 = nn.Sigmoid()
        self.act_fn_2 = nn.ReLU()

        self.BN1      = nn.BatchNorm1d(2048)
        self.BN2      = nn.BatchNorm1d(1024)
        self.BN3      = nn.BatchNorm1d(512)
        self.BN4      = nn.BatchNorm1d(256)
        self.BN5      = nn.BatchNorm1d(128)
        self.BN6      = nn.BatchNorm1d(64)

        self.dropout  = nn.Dropout(p = 0.5) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.BN1(x)
        x = self.dropout(x)
        x = self.act_fn_2(x)

        x = self.layer2(x)
        x = self.BN2(x)
        x = self.dropout(x)
        x = self.act_fn_2(x)

        x = self.layer3(x)
        x = self.BN3(x)
        x = self.act_fn_2(x)

        x = self.layer4(x)
        x = self.BN4(x)
        x = self.act_fn_2(x)

        x = self.layer5(x)
        x = self.BN5(x)
        x = self.act_fn_2(x)

        x = self.layer6(x)
        x = self.BN6(x)
        x = self.act_fn_2(x)

        x = self.out(x)

        return x

#%%
if __name__ == "__main__":
    # Use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model Path
    model_1_path = r"ensemble_model\model_1.pth" # Valid Loss : 0.712 | Public Score : 0.73703
    model_2_path = r"ensemble_model\model_2.pth" # Valid Loss : 0.712 | Public Score : 0.73557
    model_3_path = r"ensemble_model\model_3.pth" # Valid Loss : 0.547 | Public Score : 0.74603

    # Submission Path
    save_path = r"submission\ensemble_submission.csv"

    # Load Dataset
    test_data  = np.load(r'ml2021spring-hw2\test_11.npy')
    scaler = StandardScaler().fit(test_data) # Normalization
    test_data = scaler.transform(test_data)

    test_set = TIMITDataset(test_data, None, test_mode=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # Load Model
    model_1 = Net1().to(device)
    model_1.load_state_dict(torch.load(model_1_path))

    model_2 = Net2().to(device)
    model_2.load_state_dict(torch.load(model_2_path))

    model_3 = Net3().to(device)
    model_3.load_state_dict(torch.load(model_3_path))


    # Test : 
    predictions = [] # For saving all prediction

    model_1.eval() # Set the model to evaluation mode
    model_2.eval() # Set the model to evaluation mode
    model_3.eval() # Set the model to evaluation mode

    with torch.no_grad(): 
        for data in tqdm(test_loader): 
            data = data.to(device)

            pred_1 = model_1(data)
            pred_2 = model_2(data)
            pred_3 = model_3(data)

            # Ensemble : Averaging
            avg_pred = (pred_1 + pred_2 + pred_3) / 3

            # Get the position having the highest probability
            # EX : [0.2, 0.3, 0.5] => [0, 1, 2] => Get position 2
            (_, test_pred) = torch.max(avg_pred, 1)

            # Because test_pred is a matrix
            for i in test_pred.cpu().numpy():
                predictions.append(i)

    with open(save_path, 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(predictions):
            f.write('{},{}\n'.format(i, y))

    print("Save to Excel")
#%%