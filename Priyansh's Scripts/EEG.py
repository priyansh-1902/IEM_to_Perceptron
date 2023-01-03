import os
import pickle
from scipy.io import loadmat
import numpy as np

class art:

    def __init__(self):
        self.chans = None
        self.chanLabels = None
        self.chansLH = None
        self.chansRH = None
        self.artInd = None
        self.artIndCleaned = None


    
    def load_art(self, matData):

        self.chans = matData['eeg']['arf'][0][0]['chans'][0][0][0]

        chanLabels = matData['eeg']['arf'][0][0]['chanLabels'][0][0][0]
        self.chanLabels = [matData['eeg']['arf'][0][0]['chanLabels'][0][0][0][i][0] for i in range(chanLabels.size)]

        self.chansLH = matData['eeg']['arf'][0][0]['chansLH'][0][0][0]
        self.chansRH = matData['eeg']['arf'][0][0]['chansRH'][0][0][0]

        self.artInd = matData['eeg']['arf'][0][0]['artifactInd'][0][0][0]
        self.artIndCleaned = matData['eeg']['arf'][0][0]['artIndCleaned'][0][0][0]




class EEG:

    def __init__(self):
        self.eeg = None
        self.baselined = None
        self.nTrials = None
        self.preTime = None
        self.postTime = None
        self.artPreTime =  None
        self.artPostTime =  None
        
        self.artifacts_dropped = False
        self.electrodes_dropped = False
        self.art = art()
    
    def load(self, path):
        matData = loadmat(path)
        self.eeg = matData['eeg']['data'][0][0]
        self.baselined = matData['eeg']['baselined'][0][0]
        self.nTrials = matData['eeg']['nTrials'][0][0][0][0]
        self.preTime = matData['eeg']['preTime'][0][0][0][0]
        self.postTime = matData['eeg']['postTime'][0][0][0][0]
        self.artPreTime =  matData['eeg']['artPreTime'][0][0][0][0]
        self.artPostTime =  matData['eeg']['artPostTIme'][0][0][0][0]

        self.art = art()
        self.art.load_art(matData)

        
        self.eeg_evoked = matData['eeg']['evoked'][0][0]
        #self.eeg_total = matData['eeg']['total'][0][0]

    def drop_artifacts(self):
        if not self.artifacts_dropped:
            not_artInd = 1 - self.art.artInd   
            self.eeg = self.eeg[(not_artInd.transpose() == 1), :, :]
            self.eeg_evoked = self.eeg_evoked[(not_artInd.transpose() == 1), :, :]
            #self.eeg_total = self.eeg_total[(not_artInd.transpose() == 1), :, :]
            self.artifacts_dropped = True

    def drop_electrodes(self):
        if not self.electrodes_dropped:
            self.eeg = self.eeg[:, 1:21, :]
            self.eeg_evoked = self.eeg_evoked[:, 1:21, :]
            #self.eeg_total = self.eeg_total[:, 1:21, :]
            self.electrodes_dropped = True

    def eeg_total(self):
        return np.abs(self.eeg_evoked)


    def save_data(self, path):
        if not (path[-7:] == '.pickle'):
            path = path + '.pickle'
        pickle.dump(self, open(path, 'wb'))
        print(f'Successfully saved EEG data to {path}')



if __name__ == '__main__':
    
    eeg = EEG()
    eeg.load('../EEG/1_EEGfilt.mat')
    eeg.drop_artifacts()
    print(f'loaded unfiltered eeg data of shape {eeg.eeg.shape}')
    print(f'loaded filtered eeg data of shape {eeg.eeg_total().shape}')
    eeg.save_data('./1_EEG.pickle')