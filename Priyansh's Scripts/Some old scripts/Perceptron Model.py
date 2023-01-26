import numpy as np
import pickle
import time as time
import matplotlib.pyplot as plt
import sys, os
import time as t
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from EEG import EEG
from init_TF import init_TF
from behavior import load_posBin


#torch.manual_seed(1000)
eeg = EEG()
eeg.load('../EEG/1_EEGfilt.mat')
eeg.drop_artifacts().drop_electrodes()

nChans, nBins, nElectrodes = 8, 8, 20

basisSet = init_TF(nChans, nBins)

posBin = load_posBin(f'..\\data\\1_Behavior.mat')
posBin = posBin[eeg.art.artInd.transpose()==0]

assert(eeg.eeg.shape == (float(eeg.nTrials - sum(eeg.art.artInd)), nElectrodes, (float(eeg.postTime-eeg.preTime)/4)+1))


class EEGDataset(Dataset):
    def __init__(self, eeg, posBin, tf, time, start=0, end=1):
        s_i, e_i = int(start*eeg.eeg_total().shape[0]), int(end*eeg.eeg_total().shape[0])
        self.data = eeg.eeg_total()[s_i: e_i, :, (time+1000)//4]
        self.label = posBin[s_i: e_i]
        self.tf = tf
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx].astype("float32"), self.tf[self.label[idx]].astype('float32')



def data_loader(eeg, posBin, tf, time, batch_size=64):
    train_set = EEGDataset(eeg=eeg, posBin=posBin, tf=tf, time=time, start=0, end=0.7)
    val_set = EEGDataset(eeg=eeg, posBin=posBin, tf=tf, time=time, start=0.7, end=0.85)
    test_set = EEGDataset(eeg=eeg, posBin=posBin,tf=tf, time=time, start=0.85, end=1)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set))
    test_loader = DataLoader(test_set, batch_size=len(test_set))

    return train_loader, val_loader, test_loader
    
def train_perceptron(model, eeg, posBin, tf, time, start_time, batch_size=64, verbose=False):
    
    train_loader, val_loader, test_loader = data_loader(eeg, posBin, basisSet, batch_size=64, time=time)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    train_loss = []
    val_loss = []
    total_train_loss = 0.0
    epoch = 0
    while(1):
        epoch += 1
        total_train_loss = 0.0
        for labels, data in test_loader:

            out = model(data.cuda())

            loss = criterion(out, labels.cuda())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        if epoch == 1:
            train_loss.append(total_train_loss)
        elif total_train_loss<train_loss[-1]:
            train_loss.append(total_train_loss)
        else:
            break
        
        for labels, data in val_loader:
            
            out = model(data.cuda()) 
            loss = criterion(out, labels.cuda())

            val_loss.append(loss.item())
        
        if (verbose):
            print(f"Epoch {epoch}: Training Loss = {train_loss[-1]} | Validation Loss = {val_loss[-1]}")

    sys.stdout.write('\r                                                                   ')
    sys.stdout.write(f'\rTrained Model at timestamp {time} at time {t.time() - start_time} seconds')
    sys.stdout.flush()
    torch.save(model.state_dict(), f'./Trained Models/{time}.pt')
    return train_loss

class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.layer = nn.Linear(8, 20)
        torch.nn.init.xavier_normal_(self.layer.weight)
    def forward(self, x):
        return self.layer(x)