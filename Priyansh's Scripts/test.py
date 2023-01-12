import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import numpy as np
import matplotlib.pyplot as plt

class RandomDataset(Dataset):
    def __init__(self, n_examples=20, in_dim=8, out_dim=20):
        self.a = np.random.rand(n_examples, in_dim).astype("float32")
        self.b = np.random.rand(n_examples, out_dim).astype("float32")
    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, i):
        return self.a[i], self.b[i]

class Perceptron(nn.Module):
    def __init__(self, in_dim=8, out_dim=20):
        super(Perceptron, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        torch.nn.init.xavier_normal_(self.layer.weight)
    def forward(self, x):
        return self.layer(x)

def train_perceptron(epochs=100, n_examples=8, in_dim=8, out_dim=20, verbose=True):
    
    model = Perceptron().cuda()
    train_loader = DataLoader(RandomDataset(n_examples=n_examples, in_dim=in_dim, out_dim=out_dim), batch_size=n_examples)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    train_loss = []
    val_loss = []
    total_train_loss = 0.0
    epoch = 0
    while(epoch<epochs):
        epoch += 1
        total_train_loss = 0.0
        for data, labels in train_loader:
            
            out = model(data.cuda())

            loss = criterion(out, labels.cuda())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss.append(total_train_loss)
        
        if (verbose):
            print(f"Epoch {epoch}: Training Loss = {train_loss[-1]}")
        else:
            sys.stdout.write('\r                                                                   ')
            sys.stdout.write(f"\rEpoch {epoch}: Training Loss = {train_loss[-1]}")
            sys.stdout.flush()
    print(f"\nMin loss = {min(train_loss)}")
    
    plt.plot(train_loss)
    plt.show()
    return train_loss



if __name__ == '__main__':
    train_perceptron(epochs=10000, verbose=False)
    '''a = np.random.rand(8, 8).astype("float32")
    b = np.random.rand(8, 20).astype("float32")
    x = np.linalg.lstsq(a, b)
    for _ in x:
        print(_)'''