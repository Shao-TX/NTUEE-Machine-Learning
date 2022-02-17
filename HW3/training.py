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

#%%
IMAGE_SIZE = 128
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
# Set Random Seed
def Set_Seed(myseed = 41926):
    np.random.seed(myseed)

    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Plot
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
train_transform = transforms.Compose([
                                      transforms.Resize((128, 128)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5))
                                      ])

valid_transform = transforms.Compose([
                                      transforms.Resize((128, 128)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5 ,0.5))
                                      ])

train_set = ImageFolder(root = r"ml2021spring-hw3\food-11\training\labeled", transform = train_transform)
valid_set = ImageFolder(root = r"ml2021spring-hw3\food-11\validation", transform = valid_transform)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True, pin_memory=False)
valid_loader = DataLoader(valid_set, batch_size = BATCH_SIZE, shuffle=False, pin_memory=False)

#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # (in_channels, out_channels, kernel_size, stride, padding)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # CNN Layer
        x = self.cnn_layer(x)
        
        # Flatten
        # x = x.view(-1)
        x = x.flatten(1)

        # Fully Connected Layer
        x = self.fc_layers(x)

        return x

#%%
if __name__ == "__main__":
    Set_Seed()
    device = Get_Device()
    
    # Time Recording
    total_time = 0

    # Hyperparameter
    LR = 0.001
    EPOCH = 10

    # Load model
    model = Net().to(device)

    # Loss & Optimizer function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay = 0)

    # Initial Loss Record & Accuracy Record
    train_loss_record = []
    valid_loss_record = []
    train_acc_record = []
    valid_acc_record = []

    # Early Stop Parameter
    min_loss = 1000

    # Start Training
    for epoch in range(EPOCH):

        # Start Recording Time
        time_start = time.time()

        # Initial Loss & Accuracy
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        # Train :
        model.train()

        for batch in tqdm(train_loader):
            data, label = batch
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, label)
            loss.backward()
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
            for batch in tqdm(valid_loader):
                data, label = batch
                data, label = data.to(device), label.to(device)

                pred = model(data)
                loss = loss_fn(pred, label)

                # Record the Loss & Accuracy
                _, pred_index = torch.max(pred, 1)

                valid_acc  += (pred_index.cpu() == label.cpu()).sum().item()
                valid_loss += loss.item()

        # Caculate the Average Accuracy & Loss
        train_avg_acc  = (train_acc/len(train_set))*100
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

        print("Each Epoch Cost : {} s\n".format(time_cost))

    print("Total Cost Time : {} s".format(total_time))

    #%%
    # Plot the loss history
    plot_learning_curve(train_loss_record, valid_loss_record, epoch=EPOCH, title='Loss')

    # Plot the accuracy history
    plot_learning_curve(train_acc_record, valid_acc_record, epoch=EPOCH, title='Accuracy')

#%%