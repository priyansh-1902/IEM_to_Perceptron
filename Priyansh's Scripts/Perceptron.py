import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import numpy as np
import matplotlib.pyplot as plt

class RandomDataset(Dataset):
    def __init__(self, n_examples=16, in_dim=8, out_dim=20):
        self.a = np.random.rand(n_examples, in_dim).astype("float32")
        self.b = np.random.rand(n_examples, out_dim).astype("float32")*200
        
    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, i):
        return self.a[i], self.b[i]


class EEGDataset(Dataset):
    def __init__(self, B1, C1):
        self.B1 = B1
        self.C1 = C1
        
    
    def __len__(self):
        return self.B1.shape[0]

    def __getitem__(self, i):
        return self.B1[i].astype('float32'), self.C1[i].astype('float32')


class LinearPerceptron(nn.Module):
    def __init__(self, in_dim=20, out_dim=8):
        super(LinearPerceptron, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, x):
        
        #print(nn.Sigmoid()(self.layer(x)))
        return self.layer(x)

class TwoLinearPerceptron(nn.Module):
    def __init__(self, in_dim=20, out_dim=8):
        super(TwoLinearPerceptron, self).__init__()
        self.layer1 = nn.Linear(in_dim, 8)
        
        #torch.nn.init.xavier_normal_(self.layer1.weight)
        #torch.nn.init.xavier_normal_(self.layer2.weight)
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        return x

def train_perceptron(model, B1=None, C1=None, epochs=10000, random_dataset=False, verbose=False):
    
    train_loader = DataLoader(EEGDataset(B1, C1), batch_size=16)
    
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=0.2, betas=(0.9, 0.99))

    train_loss, train_acc = [], [0]
    total_train_loss = np.inf
    epoch = 0
    min_loss = np.inf
    while(total_train_loss > 0.2 and epoch<epochs):
        # if epoch==8000:
        #     optimizer.lr = 0.05
        epoch += 1
        total_train_loss = 0.0
        total_corr = 0
        total_examples = 0
        for data, labels in train_loader:
            
            optimizer.zero_grad()
            out = model(data.cuda())
            
            loss = criterion(out, labels.cuda())
            loss.backward()
            optimizer.step()
            #total_corr += sum(np.argmax(out.cpu().detach().numpy(), axis=1)==np.argmax(labels.numpy(), axis=1))
            #total_examples += out.shape[0]

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_model = model.state_dict()
                #print('Weights saved at loss ', loss.item())

            total_train_loss += loss.item()

        train_loss.append(total_train_loss)
        #train_acc.append(total_corr/total_examples)
        if (verbose):
            #sys.stdout.write('\r                                                                   ')
            sys.stdout.write(f"Epoch {epoch}: Training Loss = {train_loss[-1]}\n")#::Training Acc = {train_acc[-1]}\n")
            sys.stdout.flush()
    #print(f"\nMin loss = {min(train_loss)
    plt.show() 
    model.load_state_dict(best_model)
    return model, train_loss, train_acc, epoch


if __name__ == '__main__':
    print(train_perceptron(epochs=100, random_dataset=True, non_linearity=True, verbose=False)[1])
    '''a = np.random.rand(8, 8).astype("float32")
    b = np.random.rand(8, 20).astype("float32")
    x = np.linalg.lstsq(a, b)
    for _ in x:
        print(_)'''


        