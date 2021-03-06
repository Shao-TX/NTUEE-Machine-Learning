#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#%%
import time
import numpy as np
from sklearn.model_selection import train_test_split

# Plot
import matplotlib.pyplot as plt

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
# Set Random Seed
def Set_Seed(myseed = 41926):
    np.random.seed(myseed)

    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#%%
# Loss Plot
def plot_learning_curve(train_record, valid_record, epoch, title=''):
    epoch_len = range(epoch) # Epoch 刻度

    plt.figure(figsize=(6, 4))
    plt.plot(epoch_len, train_record, 'o-', color='red', label='train')
    plt.plot(epoch_len, valid_record, 'o-', color='blue', label='valid')

    plt.xlabel('Epoch')
    plt.ylabel(title)

    plt.xticks(epoch_len[::5]) # X 軸刻度

    plt.title('Learning Curve')
    plt.legend()
    plt.show()

#%%
# Load Data
data  = np.load(r'ml2021spring-hw2\train_11.npy')
label = np.load(r'ml2021spring-hw2\train_label_11.npy')

print("Size of training data :", data.shape)
print("Size of label         :", label.shape)

#%%
# Normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(data)
data = scaler.transform(data)


#%%
X_train, X_valid, y_train, y_valid = train_test_split(data, label, train_size=0.8,random_state=777)

y_train, y_valid = y_train.astype(np.int32), y_valid.astype(np.int32) # Beacuse CrossEntropy need Long type(Int to Long)

print("Size of Training Data :", X_train.shape)
print("Size of Valid    Data :", X_valid.shape)

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
        self.layer1   = nn.Linear(429 , 2048)
        self.layer2   = nn.Linear(2048 , 1024)
        self.layer3   = nn.Linear(1024, 512)
        self.layer4   = nn.Linear(512 , 256)
        self.layer5   = nn.Linear(256 , 128)
        self.layer6   = nn.Linear(128 , 64)
        self.out      = nn.Linear(64 , 39)

        self.act_fn_1 = nn.Sigmoid()
        self.act_fn_2 = nn.ReLU()
        self.act_fn_3 = nn.LeakyReLU()

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
        x = self.act_fn_3(x)

        x = self.layer2(x)
        x = self.BN2(x)
        x = self.dropout(x)
        x = self.act_fn_3(x)

        x = self.layer3(x)
        x = self.BN3(x)
        x = self.act_fn_3(x)

        x = self.layer4(x)
        x = self.BN4(x)
        x = self.act_fn_3(x)

        x = self.layer5(x)
        x = self.BN5(x)
        x = self.act_fn_3(x)

        x = self.layer6(x)
        x = self.BN6(x)
        x = self.act_fn_3(x)

        x = self.out(x)

        return x

#%%
if __name__ == '__main__':

    Set_Seed()
    device = Get_Device()
    
    # Time Recording
    total_time = 0

    # Hyperparameter
    LR = 0.001
    EPOCH = 250
    BATCH_SIZE = 64

    # Load model
    model = Net().to(device)

    # Loss & Optimizer function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay = 0)

    # Load Dataset
    train_set = TIMITDataset(X_train, y_train)
    valid_set = TIMITDataset(X_valid, y_valid)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

    # Initial Loss Record & Accuracy Record
    train_loss_record = []
    valid_loss_record = []
    train_acc_record = []
    valid_acc_record = []

    # Early Stop Parameter
    best_acc = 0.0
    min_loss = 1000

#%%
    # Starting Training
    for epoch in range(EPOCH):

        # Start Recording Time
        time_start = time.time()

        model.train()

        # Initial Loss & Accuracy
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0


        # Train : 
        for i, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()

            _, pred_index = torch.max(pred, 1) # get the index of the class with the highest probability

            train_acc  += (pred_index.cpu() == label.cpu()).sum().item() # 紀錄總共有幾筆資料預測正確 
            train_loss += loss.item()


        # Valid : 
        model.eval()
        with torch.no_grad():
            for i, (data, label) in enumerate(valid_loader):
                data, label = data.to(device), label.to(device)

                pred = model(data)
                loss = loss_fn(pred, label)

                _, pred_index = torch.max(pred, 1)

                valid_acc  += (pred_index.cpu() == label.cpu()).sum().item() # 紀錄總共有幾筆資料預測正確 
                valid_loss += loss.item()



            # Accuracy 是計算每筆資料所以除以 len(train_set)
            # Loss 是每個 Batch 算一次所以除以 len(train_loader)
            print('[{:03d}/{:03d}] Train Acc: {:3.2f} Loss: {:3.6f} | Val Acc: {:3.2f} loss: {:3.6f}'.format(
                epoch + 1, EPOCH, (train_acc/len(train_set))*100, train_loss/len(train_loader), (valid_acc/len(valid_set))*100, valid_loss/len(valid_loader)))

            # Save Loss Record to history
            train_loss_record.append((train_loss/len(train_loader)))
            valid_loss_record.append((valid_loss/len(valid_loader)))

            # Save Accuracy Record to history
            train_acc_record.append((train_acc/len(train_set))*100)
            valid_acc_record.append((valid_acc/len(valid_set))*100)

            # if the model improves, save a checkpoint at this epoch
            if((valid_loss/len(valid_loader)) < min_loss):
                min_loss = (valid_loss/len(valid_loader))
                torch.save(model.state_dict(), 'model.pth')
                print('Saving model with loss {:.3f}'.format(min_loss)) # 可以改 loss 作為 early_stop
            
        time_end = time.time()              # Finish Recording Time
        time_cost = time_end - time_start   # Time Spent
        total_time = total_time + time_cost # Total Time
        
        print("Each Epoch Cost : {} s\n".format(time_cost))

print("Total Cost Time : {} s".format(total_time))


#%%
# Plot the loss history
plot_learning_curve(train_loss_record, valid_loss_record, epoch=EPOCH, title='Loss')

# Plot the accuracy history
plot_learning_curve(train_acc_record, valid_acc_record, epoch=EPOCH, title='Accuracy')

#%%