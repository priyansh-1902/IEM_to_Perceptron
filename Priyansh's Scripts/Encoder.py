import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


import numpy as np
import matplotlib.pyplot as plt

class RandomDataset(Dataset):
    def __init__(self, n_examples=802, in_dim=20, out_dim=8):
        self.a = np.random.rand(n_examples, in_dim).astype("float32")
        self.b = np.random.rand(n_examples, out_dim).astype("float32")
        
    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, i):
        return self.a[i], self.b[i]


class EEGDataset(Dataset):
    def __init__(self, B1, C1, posBin):
        self.B1 = B1
        self.C1 = C1
        self.posBin = posBin
    def __len__(self):
        return self.B1.shape[0]
    def __getitem__(self, i):
        return self.B1[i].astype('float32'), self.posBin[i]

class LinearEncoder(nn.Module):
    def __init__(self, in_dim=20, out_dim=8):
        super(LinearEncoder, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.xavier_normal_(self.layer.weight)
    def forward(self, x):
        return nn.Softmax()(self.layer(x))

class NonLinearEncoder(nn.Module):
    def __init__(self, in_dim=20, out_dim=8):
        super(NonLinearEncoder, self).__init__()
        self.layers = nn.Sequential(nn.Linear(20, 16),
                                    nn.Tanh(),
                                    nn.Linear(16, 12),
                                    nn.ReLU(),
                                    nn.Linear(12, 8)
                                    
                                    )
        
    def forward(self, x):
        return self.layers(x)



def train_encoder(train_eeg=None, train_basis_set=None, test_eeg=None, posBin=None, batch_size=64, epochs=1000, non_linearity=False, random_dataset=False, verbose=False):
    
    if (non_linearity):
        model = NonLinearEncoder().cuda()
    else:
        model = LinearEncoder().cuda()

    if random_dataset:
        train_loader = DataLoader(RandomDataset(), batch_size=16)
    else:
        train_loader = DataLoader(EEGDataset(B1=train_eeg, C1=train_basis_set, posBin=posBin), batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=6e-3, betas=(0.9, 0.99))

    train_loss = []
    total_train_loss = 0.0
    epoch = 0
    min_loss = np.inf
    while(epoch<epochs):
        epoch += 1
        total_train_loss = 0.0
        total_corr = 0
        total_examples = 0
        for data, labels in train_loader:

            optimizer.zero_grad()

            out = model(data.cuda()).squeeze()
            
            loss = criterion(out, labels.cuda())
            loss.backward()
            optimizer.step()
            total_corr += sum(torch.argmax(out, axis=1).cpu()==labels)
            total_examples += labels.shape[0]

            total_train_loss += loss.item()

        if total_train_loss < min_loss:
                min_loss = total_train_loss
                best_model = model.state_dict()
                #print('Weights saved at loss ', total_train_loss, 'at epoch ', epoch)

        train_loss.append(total_train_loss)
        
        if (verbose):
            sys.stdout.write('\r                                                                   ')
            sys.stdout.write(f"\rEpoch {epoch}: Training Loss = {train_loss[-1]}, Training Accuracy = {total_corr/total_examples}\n")
            sys.stdout.flush()

        if total_corr==total_examples:
            break
    #print(f"\nMin loss = {min(train_loss)}")
    plt.plot(train_loss)
    plt.show() 
    model.load_state_dict(best_model)
    return model(torch.Tensor(test_eeg).cuda()).detach().cpu()


if __name__=='__main__':
    train_encoder(epochs=300, random_dataset=True)