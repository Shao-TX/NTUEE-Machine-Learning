#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from torchvision.datasets import ImageFolder
from torchsummary import summary

# Tool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

#%%
IMAGE_SIZE = 128
BATCH_SIZE = 128

#%%
def Get_Device():
    if(torch.cuda.is_available()):
        print("Device : GPU")
        device = torch.device("cuda:0")
    else:
        print("Device : CPU")
        device = torch.device("cpu")

    return device

#%%
# Set Random Seed
def Set_Seed(myseed = 2022):
    np.random.seed(myseed)

    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Plot
def plot_learning_curve(train_record, valid_record, epoch, title=''):
    epoch_len = range(epoch) # Epoch 刻度
    x = int(epoch/10) # 將 Epoch 刻度總間距除以 10 (ex : EPOCH = 100, 間距會 = 10), int 是因為 list 需要整數

    plt.figure(figsize=(6, 4))
    plt.plot(epoch_len, train_record,  color='red', label='train')
    plt.plot(epoch_len, valid_record,  color='blue', label='valid')

    plt.xlabel('Epoch')
    plt.ylabel(title)

    plt.xticks(epoch_len[::x]) # X 軸刻度

    plt.title('Learning Curve of ' + title)
    plt.legend()
    plt.show()

#%%
train_transform = transforms.Compose([
                                      transforms.ColorJitter(brightness=(0.5, 1.2)),                          # 隨機亮度調整
                                    #   transforms.RandomHorizontalFlip(p=0.5),                                 # 隨機水平翻轉
                                      transforms.RandomRotation((-40, 40)),                                   # 隨機旋轉
                                      transforms.RandomResizedCrop(size = IMAGE_SIZE, scale = (0.5, 1.5)),    # 隨機縮放
                                      
                                      transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5))
                                      ])

valid_transform = transforms.Compose([
                                      transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5))
                                      ])

train_set = ImageFolder(root = r"ml2021spring-hw3\food-11\training\new_labeled", transform = train_transform)
valid_set = ImageFolder(root = r"ml2021spring-hw3\food-11\validation", transform = valid_transform)
semi_set  = ImageFolder(root = r"ml2021spring-hw3\food-11\training\unlabeled", transform = valid_transform)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size = BATCH_SIZE, shuffle=False, pin_memory=True)

#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # (in_channels, out_channels, kernel_size, stride, padding)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # 128 -> 64

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # 64 -> 32

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # 32 -> 16

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # 16 -> 8

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # 8 -> 4
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.7),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.7),
            nn.ReLU(),

            # nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            # nn.Dropout(0.5),
            # nn.ReLU(),

            nn.Linear(512, 11)
        )

    def forward(self, x):
        # CNN Layer
        x = self.cnn_layer(x)
        
        # Flatten
        x = x.flatten(1)

        # Fully Connected Layer
        x = self.fc_layers(x)

        return x

#%%
class PseudoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, id):
        return self.x[id][0], self.y[id] # x[id][0] : 只取原 dataset 中的 data 不取 label, y[id] : new label

def get_pseudo_labels(dataset, model, batch_size, device, threshold=0.90):

    data_loader = DataLoader(dataset, batch_size, shuffle=False)

    model.eval()
    softmax = nn.Softmax(dim=-1)

    idx = [] # 紀錄哪些位置的 data 要用於訓練
    labels = [] # 哪些位置的 data 的 label

    i = 0 # 用於計數 : idx
    for (img, _) in tqdm(data_loader):
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)
        
        for j, x in enumerate(probs):
            if torch.max(x) > threshold:
                
                idx.append(i * batch_size + j) # EX : batch = 128, i = 0, j = 0 : idx = 0 * 128 + 0 = 0
                labels.append(int(torch.argmax(x)))
        
        i += 1

    model.train()
    print ("\nNew data: {:5d}\n".format(len(idx)))
    dataset = PseudoDataset(Subset(dataset, idx), labels) # Subset : 取特定 index 的 data
    
    return dataset

#%%
if __name__ == "__main__":
    Set_Seed()
    device = Get_Device()
    
    # Time Recording
    total_time = 0

    # Hyperparameter
    LR = 1e-5
    EPOCH = 300

    # Load model
    model = Net()
    model.to(device)

    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))

    # Loss & Optimizer function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay = 1e-4)

    # Initial Loss Record & Accuracy Record
    train_loss_record = []
    valid_loss_record = []
    train_acc_record = []
    valid_acc_record = []

    # Semi
    do_semi = True

    # Initial Some Value
    min_loss = 1000      # 用於判定是否該存 model
    valid_avg_acc = 0    # 用於判定是否該啟用 Semi-Supervised Learning

    # Start Training
    for epoch in range(EPOCH):

        # Start Recording Time
        time_start = time.time()

        # Initial Loss & Accuracy
        train_loss = 0.0
        valid_loss = 0.0
        train_acc  = 0.0
        valid_acc  = 0.0

        # 當 Valid Accuracy > 60% 時才加入 Unlabeled Data
        if(do_semi and (valid_avg_acc > 60)):
            # Obtain pseudo-labels for unlabeled data using trained model.
            pseudo_set = get_pseudo_labels(semi_set, model, BATCH_SIZE, device)

            # This is used in semi-supervised learning only.
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True) # 要開啟 drop_last : 如果剛好資料剩一筆，因為 BatchNormalize 要大於 1 筆才運作

            # For caculate the train_acc
            train_acc_len = len(concat_dataset)
        else : 
            # For caculate the train_acc
            train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)
            train_acc_len = len(train_set)

        # Train :
        model.train()

        for (data, label) in tqdm(train_loader):
            data, label = data.to(device), label.to(device)

            pred = model(data)
            loss = loss_fn(pred, label)   
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10) # 梯度剪裁
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (pred.argmax(dim=-1) == label.to(device)).float().mean()

            # Record the Loss & Accuracy
            _, pred_index = torch.max(pred, 1) # get the index of the class with the highest probability

            train_acc  += (pred_index.cpu() == label.cpu()).sum().item() # [1, 0, 1] => mean = (1 + 0 + 1)/3 = 0.66
            train_loss += loss.item()

        # Valid :
        model.eval()

        with torch.no_grad():
            for (data, label) in tqdm(valid_loader):
                data, label = data.to(device), label.to(device)

                pred = model(data)
                loss = loss_fn(pred, label)

                # Record the Loss & Accuracy
                _, pred_index = torch.max(pred, 1)

                valid_acc  += (pred_index.cpu() == label.cpu()).sum().item()
                valid_loss += loss.item()

            # Caculate the Average Accuracy & Loss
            train_avg_acc  = (train_acc/train_acc_len)*100
            valid_avg_acc  = (valid_acc/len(valid_set))*100
            
            train_avg_loss = train_loss/len(train_loader)
            valid_avg_loss = valid_loss/len(valid_loader)

            # Show the Accuracy & Loss each epoch
            print('[{:03d}/{:03d}] Train Acc: {:3.2f} Loss: {:3.6f} | Val Acc: {:3.2f} loss: {:3.6f}'.format(epoch + 1, EPOCH, train_avg_acc, train_avg_loss, valid_avg_acc, valid_avg_loss))

            # if the model improves, save a checkpoint at this epoch
            if(valid_avg_loss < min_loss):
                min_loss = (valid_loss/len(valid_loader))
                torch.save(model.state_dict(), 'model.pth')
                print('Saving model with loss {:.3f}'.format(min_loss))

            # Save Accuracy Record to history(Plot)
            train_acc_record.append(train_avg_acc)
            valid_acc_record.append(valid_avg_acc)

            # Save Loss Record to history(Plot)
            train_loss_record.append(train_avg_loss)
            valid_loss_record.append(valid_avg_loss)

        time_end = time.time()              # Finish Recording Time
        time_cost = time_end - time_start   # Time Spent
        total_time = total_time + time_cost # Total Time

        print("Each Epoch Cost : {:3.3f} s\n".format(time_cost))

    print("Total Cost Time : {:3.3f} s".format(total_time))

    # Plot the loss history
    plot_learning_curve(train_loss_record, valid_loss_record, epoch=EPOCH, title='Loss')

    # Plot the accuracy history
    plot_learning_curve(train_acc_record, valid_acc_record, epoch=EPOCH, title='Accuracy')

#%%