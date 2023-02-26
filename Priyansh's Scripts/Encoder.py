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
        self.layers = nn.Sequential(nn.BatchNorm1d(20),
                                    nn.ReLU(),
                                    nn.Linear(20, 16),
                                    nn.Sigmoid(),
                                    nn.Linear(16, 14),
                                    nn.ReLU(),
                                    nn.Linear(14, 12),
                                    nn.ReLU(),
                                    nn.Linear(12, 10),
                                    nn.ReLU(),
                                    nn.Linear(10, 8))
        
    def forward(self, x):
        return self.layers(x)



def train_encoder(train_eeg, train_basis_set, test_eeg, train_posBin, test_posBin, batch_size=32, epochs=1500, non_linearity=False, verbose=False):
    
    if (non_linearity):
        model = NonLinearEncoder().cuda()
    else:
        model = LinearEncoder().cuda()

    train_loader = DataLoader(EEGDataset(B1=train_eeg, C1=train_basis_set, posBin=train_posBin), batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-3)

    train_loss, train_acc, test_acc = [], [], []
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
        train_acc.append(total_corr/total_examples)

        test_out = model(torch.Tensor(test_eeg).cuda()).detach().cpu()
        
        test_corr = sum(torch.argmax(test_out, axis=1).cpu().numpy()==test_posBin)
        test_acc.append(test_corr/test_out.shape[0])
        
        if (verbose):
            sys.stdout.write('\r                                                                   ')
            sys.stdout.write(f"\rEpoch {epoch}: Traini Loss = {train_loss[-1]}, Train loss = {train_acc[-1]}, Test acc = {test_acc[-1]}\n")
            sys.stdout.flush()
    #print(f"\nMin loss = {min(train_loss)}")
    fig, ax = plt.subplots(2)
    ax[0].plot(train_loss)
    ax[0].set_title('Training Loss')
    ax[1].plot(train_acc)
    ax[1].plot(test_acc)
    ax[1].legend(['Training accuracy', 'Test accuracy'])
    ax[1].set_ylim([0,1])
    ax[1].set_title('Training Accuracy')
    plt.show() 
    model.load_state_dict(best_model)
    return model(torch.Tensor(test_eeg).cuda()).detach().cpu()


if __name__=='__main__':
    train_encoder(epochs=300, random_dataset=True)