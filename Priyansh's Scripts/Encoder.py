import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import numpy as np
import matplotlib.pyplot as plt

class RandomDataset(Dataset):
    def __init__(self, n_examples=16, in_dim=20, out_dim=8):
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


class LinearEncoder(nn.Module):
    def __init__(self, in_dim=20, out_dim=8):
        super(LinearEncoder, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.xavier_normal_(self.layer.weight)
    def forward(self, x):
        return self.layer(x)

class NonLinearEncoder(nn.Module):
    def __init__(self, in_dim=8, out_dim=20):
        super(NonLinearEncoder, self).__init__()
        self.layer1 = nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.xavier_normal_(self.layer.weight)
    def forward(self, x):
        return torch.nn.ReLU()(self.layer(x))

def train_encoder(B1=None, C1=None, B2=None, C2=None, epochs=100, non_linearity=False, random_dataset=False, verbose=False):
    
    if (non_linearity):
        model = NonLinearEncoder().cuda()
    else:
        model = LinearEncoder().cuda()

    if random_dataset:
        train_loader = DataLoader(RandomDataset(), batch_size=16)
    else:
        train_loader = DataLoader(EEGDataset(B1, C1), batch_size=B1.shape[0])
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.00005)

    train_loss = []
    total_train_loss = 0.0
    epoch = 0
    min_loss = np.inf
    while(epoch<epochs):
        epoch += 1
        total_train_loss = 0.0
        for data, labels in train_loader:
            
            out = model(data.cuda())

            loss = criterion(out, labels.cuda())
            loss.backward()
            optimizer.step()

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_weights = [model.state_dict()[param_tensor] for param_tensor in model.state_dict()][0].cpu().numpy().squeeze()
                print('Weights saved at loss ', loss.item())

            total_train_loss += loss.item()

        train_loss.append(total_train_loss)
        
        if (verbose):
            #sys.stdout.write('\r                                                                   ')
            sys.stdout.write(f"Epoch {epoch}: Training Loss = {train_loss[-1]}\n")
            sys.stdout.flush()
    #print(f"\nMin loss = {min(train_loss)}")
    plt.plot(train_loss)
    plt.show() 

    return best_weights.transpose()

if __name__=='__main__':
    train_encoder(epochs=300, random_dataset=True)