import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import numpy as np


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
        return self.layer(x)

class NonLinearPerceptron(nn.Module):
    def __init__(self, in_dim=20, out_dim=8):
        super(NonLinearPerceptron, self).__init__()
        self.layer1 = nn.Linear(in_dim, 8)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        return x
    

class NonLinearMLP(nn.Module):
    def __init__(self):
        super(NonLinearMLP, self).__init__()
        self.layer1 = nn.Linear(20, 14)
        self.layer2 = nn.Linear(14, 8)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Sigmoid()(x)
        return x

def train_perceptron_cpu(_model, B1=None, C1=None, max_epochs=300, verbose=False):
    
    model = _model()
    train_loader = DataLoader(EEGDataset(B1, C1), batch_size=16)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.2, betas=(0.9, 0.99))

    train_loss = []
    total_train_loss = np.inf
    epoch = 0
    min_loss = np.inf
    while(total_train_loss > 0.2 and epoch<max_epochs):
        epoch += 1
        total_train_loss = 0.0

        for data, labels in train_loader:
            
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_model = model.state_dict()

            total_train_loss += loss.item()

        train_loss.append(total_train_loss)

        if (verbose):
            sys.stdout.write(f"Epoch {epoch}: Training Loss = {train_loss[-1]}\n")

    model.load_state_dict(best_model)
    return model, train_loss, epoch


if __name__ == '__main__':
    print(train_perceptron_cpu(epochs=100, random_dataset=True, non_linearity=True, verbose=False)[1])


        