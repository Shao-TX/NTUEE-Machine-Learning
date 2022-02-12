#%%
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Data Preprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#%%
# 存成 Kaggle 形式
def save_pred(preds, df, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    for i, p in enumerate(preds):
        i+=1 # Pandas 是從 1 開始(但 i 等於 0)
        df.loc[i] = [i-1, p]
        print("test")

    df['id'] = df['id'].astype('Int32') # 因為 csv 是由 String 存入，因此需轉成 Int32 才能上傳 Kaggle
    df.to_csv(file + ".csv",
            index = False)

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
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1   = nn.Linear(429 , 1024)
        self.layer2   = nn.Linear(1024, 512)
        self.layer3   = nn.Linear(512 , 128)
        self.out      = nn.Linear(128 , 39)

        self.act_fn_1 = nn.Sigmoid()
        self.act_fn_2 = nn.ReLU()

        self.BN1      = nn.BatchNorm1d(1024)
        self.BN2      = nn.BatchNorm1d(512)
        self.BN3      = nn.BatchNorm1d(128)

        self.dropout  = nn.Dropout(p = 0.5) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.BN1(x)
        x = self.act_fn_2(x)

        x = self.layer2(x)
        x = self.BN2(x)
        x = self.act_fn_2(x)

        x = self.layer3(x)
        x = self.BN3(x)
        x = self.act_fn_2(x)

        x = self.out(x)

        return x

#%%
if __name__ == "__main__":
    # Use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model Path
    model_name = input("Enter model name")
    model_path = model_name + ".pth"
    save_path = model_name + "_submission.csv"

    # Load Dataset
    test_data  = np.load(r'ml2021spring-hw2\test_11.npy')
    scaler = StandardScaler().fit(test_data)
    test_data = scaler.transform(test_data)

    test_set = TIMITDataset(test_data, None, test_mode=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

#%%
    # Load Model
    model = Net().to(device)

    # Load Best Model Weight
    model.load_state_dict(torch.load(model_path))


    # Test
    model.eval()
    preds = []
    with torch.no_grad(): 
        for i, data in enumerate(test_loader): 
            data = data.to(device)
            pred = model(data)

            _, test_pred = torch.max(pred, 1)
            preds.append(test_pred.detach().cpu().numpy().item())

#%%
    with open(save_path, 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(preds):
            f.write('{},{}\n'.format(i, y))

#%%