#%%
# Pytorch
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
# Set Random Seed
def set_seed(myseed = 45215):
    np.random.seed(myseed)

    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
# 畫出學習曲線
def plot_learning_curve(loss_record, title=''):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['valid'])] # Training Data 整筆資料比 Valid Data 還多
    
    # train 總資料量 / test 總資料量
    print('loss_record [train] : ', len(loss_record['train']))
    print('loss_record [valid] : ', len(loss_record['valid']))

    plt.figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['valid'], c='tab:cyan', label='valid')
    plt.ylim(0.0, 10.0) # 因為一開始 error 會很大，必須限制圖表範圍，否則圖像會失真
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

# 畫出預測曲線

def plot_pred(valid_data, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []

        for data, target in valid_data:

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
    def __init__(self, path, mode = 'train',target_only=True):
        self.mode = mode

        data = pd.read_csv(path)
        data = data.iloc[:, 1:] # 不取編號(第一行)
        data = np.array(data) # Pandas to Numpy

        if not target_only:
            feats = list(range(93))
        else:
            feats = [40, 41, 42, 43, 57, 58, 59, 60, 61, 75, 76, 77, 78, 79] # Feature Select

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
        print()
        # Normalize : (X - Mean) / Std
        # self.data[:, :] = (self.data[:, :] - self.data[:, :].mean(dim=0, keepdim=True)) / self.data[:, :].std(dim=0, keepdim=True) # dim => 0:行運算, 1:列運算
        
        self.dim = self.data.shape[1] # 93 => data = [Number of data, 93]

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
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=True):
    dataset = COVID19Dataset(path, mode = mode, target_only = target_only)  # Construct dataset
    
    dataloader = DataLoader(
                            dataset, 
                            batch_size,
                            shuffle=(mode == 'train'), # (1) Train : True (2) Vaild & Test : False
                            drop_last=True,
                            num_workers=n_jobs, 
                            pin_memory=True
                        )
    
    return dataloader

#%%
# Create Neural Networks
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.main(x).squeeze(1) # Out = ([270, 1]),  但 Target 為 ([270]) => 需降維(PS:也可.view(-1))


#%%
# Validation
def valid(valid_data, model, device):
    loss_fn = nn.MSELoss()
    model.eval()                                                 # set model to evalutation mode
    total_loss = 0
    for data, target in valid_data:                              # iterate through the dataloader

        data = data.to(device)
        target = target.to(device)                               # move data to device (cpu/cuda)

        with torch.no_grad():                                    # disable gradient calculation
            pred = model(data)                                   # forward pass (compute output)
            loss = loss_fn(pred, target)                         # compute loss
        
        total_loss += loss.detach().cpu().item() * len(data)     # accumulate loss
    total_loss = total_loss / len(valid_data.dataset)            # compute averaged loss

    return total_loss

#%%
# Training
def train(train_data, valid_data, model, epoch, device, saving_name):

    # Setup Loss Function
    loss_fn = nn.MSELoss()

    # Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.01) # 加入 L2 Regulization(必定使 training loss 變差)

    min_loss = 1000 # 初始化最小 loss
    loss_record = {'train': [], 'valid': []}      # For recording training loss
    early_stop_cnt = 0

    saving_name = saving_name + '.pth'
    for i in range(epoch):
        model.train()                             # set model to training mode

        for data, target in train_data:           # iterate through the dataloader

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
        valid_loss = valid(valid_data, model, device)

        # Early Stop : 比上一次最小的 loss 還小才存 model
        if valid_loss < min_loss:
            min_loss = valid_loss # 若小於上一筆最小的 loss 就用此 loss 取代
            print('Saving model (epoch = {:4d}, loss = {:.4f})'.format(i + 1, min_loss))
            torch.save(model.state_dict(), saving_name) # 因為 loss 變小了，因此將此 model 存下
            early_stop_cnt = 0 # 因為 loss 被重置為新的，所以 early stop 步數也重置
        else:
            early_stop_cnt += 1

        loss_record['valid'].append(valid_loss)
        if early_stop_cnt > 200: # 超過 200 次沒有比前面的 loss 還小就停止 training
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_loss, loss_record

#%%
if __name__ == "__main__" :
    # Set Random Seed
    set_seed()

    # Use GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data Path
    TRAIN_PATH = 'ml2021spring-hw1\covid.train.csv'  # path of training data
    
    # Batch Size
    BATCH_SIZE = 128

    # Epoch
    EPOCH = 3000

    # Load Dataset
    train_data = prep_dataloader(TRAIN_PATH, mode = 'train', batch_size = BATCH_SIZE, target_only = True)
    valid_data = prep_dataloader(TRAIN_PATH, mode = 'valid', batch_size = BATCH_SIZE, target_only = True)

    # Load Model
    model = Net(train_data.dataset.dim).to(device)
    print(model)

    # Saving Name
    saving_name = input("Input your model saving name : ")

    # Start Training
    model_loss, model_loss_record = train(train_data, 
                                          valid_data, 
                                          model, 
                                          epoch = EPOCH, 
                                          device = device, 
                                          saving_name = saving_name
                                          )


    # Plot Laerning Curve & Valid Prediction
    plot_learning_curve(model_loss_record, title='deep model')
    plot_pred(valid_data, model, device)

#%%