#%%
# Pytorch
from operator import index
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Data Preprocess
import numpy as np
import pandas as pd

# Plot
import matplotlib.pyplot as plt

#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%%
def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])] # (0, 27000, 9) => len(loss_record['train']) // len(loss_record['dev']) == 9

    plt.figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []

        for data, target in dv_set:

            data = data.to(device)
            target = target.to(device)

            with torch.no_grad():
                pred = model(data)
                preds.append(pred.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())

    plt.figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([0, lim], [0, lim], c='b')
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

#%%
# Dataset
class COVID19Dataset(Dataset):
    def __init__(self, path, mode = 'train',target_only=False):
        self.mode = mode

        data = pd.read_csv(path)
        data = data.iloc[:, 1:] # 不取編號(第一行)
        data = np.array(data) # Pandas to Numpy


        feats = list(range(93))


        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]

            self.data = torch.FloatTensor(data) # Convert data into PyTorch tensors

        else:
            # Training data (train/valid sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1] # day 3 的 target
            data = data[:, feats] # 2700 x 93
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 15 != 0] # 2430 筆
            elif mode == 'valid':
                indices = [i for i in range(len(data)) if i % 15 == 0] # 270 筆

            self.data = torch.FloatTensor(data[indices]) # Convert data into PyTorch tensors
            self.target = torch.FloatTensor(target[indices])

        # Normalize : (X - Mean) / Std
        self.data[:, 40:] = (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) / self.data[:, 40:].std(dim=0, keepdim=True)
        
        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'valid']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]


    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

#%%
# DataLoader
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    
    dataloader = DataLoader(
                            dataset, 
                            batch_size,
                            shuffle=(mode == 'train'), # 如果使用 train 就 True 否則 False
                            drop_last=True,
                            num_workers=n_jobs, 
                            pin_memory=True
                        ) # Construct dataloader
    
    return dataloader

#%%
class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.main(x).squeeze(1) # Out = ([270, 1]),  但 Target 為 ([270]) => 需降維(PS:也可.view(-1))

#%%
# Training

def train(tr_set, dv_set, model, device):

    Epoch = 3000 # Maximum number of epochs

    # Setup optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.01) # 加入 L2 Regulization(必定使 training loss 變差)

    min_mse = 1000
    loss_record = {'train': [], 'dev': []}        # for recording training loss
    early_stop_cnt = 0
    epoch = 0

    for i in range(Epoch):
        model.train()                             # set model to training mode

        for data, target in tr_set:               # iterate through the dataloader

            data = data.to(device)                # move data to device (cpu/cuda)
            target = target.to(device)

            optimizer.zero_grad()                 # set gradient to zero
            pred = model(data)                    # forward pass (compute output)
            loss = loss_fn(pred, target)          # compute loss
            loss.backward()                       # compute gradient (backpropagation)
            optimizer.step()                      # update model with optimizer

            loss_record['train'].append(loss.detach().cpu().item()) 
            # detach : 阻斷 Backpropagation
            # cpu : 有些動作需在 cpu 中才能執行(ex : numpy)
            # item : 取出 torch 中的值


        # Validition
        dev_mse = dev(dv_set, model, device)

        # 比上一筆 loss 小才存 model
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(epoch + 1, min_mse))
            torch.save(model.state_dict(), 'model.pth')  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > 200: # 超過 500 次沒有筆前面的 loss 還小就停止 training
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

#%%
# Validation

def dev(dv_set, model, device):
    loss_fn = nn.MSELoss()
    model.eval()                                                 # set model to evalutation mode
    total_loss = 0
    for data, target in dv_set:                                  # iterate through the dataloader

        data = data.to(device)
        target = target.to(device)                               # move data to device (cpu/cuda)

        with torch.no_grad():                                    # disable gradient calculation
            pred = model(data)                                   # forward pass (compute output)
            loss = loss_fn(pred, target)                         # compute loss
        
        total_loss += loss.detach().cpu().item() * len(data)     # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)                # compute averaged loss

    return total_loss

#%%
# Testing

def test(tt_set, model, device):
    model.eval()                                         # set model to evalutation mode
    preds = []
    for x in tt_set:                                     # iterate through the dataloader

        x = x.to(device)                                 # move data to device (cpu/cuda)

        with torch.no_grad():                            # disable gradient calculation
            pred = model(x)                              # forward pass (compute output)
            preds.append(pred.detach().cpu().numpy().item())   # collect prediction
    return preds

#%%
# Load Dataset
tr_path = 'ml2021spring-hw1\covid.train.csv'  # path to training data
tt_path = 'ml2021spring-hw1\covid.test.csv'   # path to testing data

tr_set = prep_dataloader(tr_path, 'train', batch_size = 128, target_only = False)
dv_set = prep_dataloader(tr_path, 'valid', batch_size = 128, target_only = False)
tt_set = prep_dataloader(tt_path, 'test',  batch_size = 1, target_only = False)

# Load Model
model = Model(tr_set.dataset.dim).to(device)
print(model)

#%%
# Start Training
model_loss, model_loss_record = train(tr_set, dv_set, model, device)

#%%
plot_learning_curve(model_loss_record, title='deep model')

#%%
plot_pred(dv_set, model, device)
#%%
# 測試 Testing Data
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
del model
model = Model(tt_set.dataset.dim).to(device)
ckpt = torch.load('model.pth', map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)

df = pd.DataFrame([], columns = ['id', 'tested_positive'])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, df,'submission')         # save prediction file to pred.csv

#%%