import numpy as np
from scipy.io import loadmat
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
        

    def drop_artifacts(self):
        if not self.artifacts_dropped:
            not_artInd = 1 - self.art.artIndCleaned  
            self.eeg = self.eeg[(not_artInd.transpose() == 1), :, :]
            self.eeg_evoked = self.eeg_evoked[(not_artInd.transpose() == 1), :, :]
            self.nTrials -= self.eeg_evoked.shape[0]
            self.artifacts_dropped = True
        return self

    def drop_electrodes(self):
        if not self.electrodes_dropped:
            self.eeg = self.eeg[:, 1:21, :]
            self.eeg_evoked = self.eeg_evoked[:, 1:21, :]
            #self.eeg_total = self.eeg_total[:, 1:21, :]
            self.electrodes_dropped = True
        return self

    def eeg_total(self):
        return np.square(np.abs(self.eeg_evoked))


    def save_data(self, path):
        if not (path[-7:] == '.pickle'):
            path = path + '.pickle'
        pickle.dump(self, open(path, 'wb'))
        print(f'Successfully saved EEG data to {path}')




def mexican_hat():
    import scipy
    vec = scipy.signal.ricker(100, 20)
    idx = np.array([43, 57, 71, 85, 99, 0, 14, 28])
    return [0, 0, -0.4, 0.8, -0.4, 0, 0, 0]

def init_mexican_hat_tf():
    basisSet = np.empty((8, 8,))

    for c in range(8):
        basisSet[c-1, : ] = np.roll(mexican_hat(), -c)

    return basisSet

def plot_pdf():
    
    import scipy.stats as stats
    import math

    mu = 4
    variance = 1.5
    sigma = math.sqrt(variance)
    x = np.linspace(0, 8, 9)
    y = 2.5*stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y)
    
    b = np.hstack((init_TF(8,8), init_TF(8,8)[:, 0][:, np.newaxis]))
    plt.plot(b[4])
    plt.legend(['Gaussian', 'Sinusoid'])
    plt.show()

def init_TF(nChans, nBins):
    sinPower = 7
    x = np.linspace(0, 2 * np.pi - 2 * np.pi / nBins, nBins)
    cCenters = np.linspace(0, 2 * np.pi - 2 * np.pi / nChans, nChans)
    cCenters = np.rad2deg(cCenters)
    pred = np.abs(np.sin(0.5 * x) ** sinPower) # hypothetical channel responses
    pred = np.roll(pred, -3)  # shift the initial basis function
    basisSet = np.empty((nChans, nBins,))
    for c in range(nChans):
        basisSet[c - 1, :] = np.roll(pred,-c)   

    return basisSet


def load_pos_bin(path):
    pos = loadmat(path)['ind']['cueBin'][0][0].transpose()
    pos = pos - 1
    return pos.squeeze()


def load_eeg_and_bin(eeg_path, pos_bin_path, 
                    drop_electrodes=True, drop_artifacts=True):
    
    eeg = EEG()
    eeg.load(eeg_path)

    #may need to drop artifacts and electrodes

    posBin = load_pos_bin(pos_bin_path)
    
    if drop_artifacts:
        not_artInd = 1 - eeg.art.artIndCleaned
        posBin = posBin[(not_artInd.transpose()==1)]

    return eeg, posBin


def make_blocks(tois, eeg, posBin, nBlocks, nTrials, nBins=8):
    
    # ----------------------------------------------------------
    # Assign trials to blocks (such that trials per position are
    # equated within blocks)
    # -----------------------------------------------------------

    toi_start, toi_end = tois
    
    binCnt = np.empty((nBins))
    for bin in range(nBins):
        binCnt[bin - 1] = np.sum(1 * np.equal(posBin, bin))
    # Data from binCnt: binCnt[0, x] where 0 <= x <= 7

    minCnt = min(binCnt)  # No. of trials for position bin with fewest trials
    
    # max No. of trials such that the No. of trials for each bin can be equated within each block
    nPerBin = int(minCnt / nBlocks)

    # shuffle trials
    shuffInd = np.array([np.random.permutation(nTrials)]).transpose()#np.arange(nTrials)#np.array([np.random.permutation(nTrials)]).transpose()  # create shuffle index
    
    shuffBin = posBin[shuffInd]  # shuffle trial order
    shuffeeg_evoked = eeg.eeg_evoked[shuffInd.squeeze(), :, :]
    shuffeeg_total = eeg.eeg_total()[shuffInd.squeeze(), :, :]

    # Data from shuffInd and ShuffBin: shuff[x, 0] where 0 <= x <= 119
                
    blockDat_total = []
    blockDat_evoked = []
    for block in range(nBlocks):
        blockDat_total.append([])
        blockDat_evoked.append([])
        for i in range(nBins):
            
            binSpecific_evoked = shuffeeg_evoked[(shuffBin==i).squeeze(), :, :]
            binSpecific_total = shuffeeg_total[(shuffBin==i).squeeze(), :, :]

            average_evoked = np.mean(np.square(abs(binSpecific_evoked[nPerBin*block:nPerBin*(block+1)])), axis=0) 
            average_total = np.mean(binSpecific_total[nPerBin*block:nPerBin*(block+1)], axis=0)
            
            blockDat_evoked[-1].append(average_evoked)
            blockDat_total[-1].append(average_total)
            
    blockDat_total = np.array(blockDat_total)
    blockDat_evoked = np.array(blockDat_evoked)

    return blockDat_total

    
def make_train_test_sets(tois, eeg, basisSet, posBin, nTrials, split_ratio=0.9, nBins=8):
    toi_start, toi_end = tois
    times = np.array([range(toi_start, toi_end, 4)]).squeeze()

    eeg_total = eeg.eeg_total()[:, :, (times+1000)//4]
    
    # shuffle trials
    shuffInd = np.array([np.random.permutation(nTrials)]).transpose()#np.arange(nTrials)#np.array([np.random.permutation(nTrials)]).transpose()  # create shuffle index
    
    train_posBin = posBin[shuffInd[:int(split_ratio*nTrials)].squeeze()]  # shuffle trial order
    train_eeg_total = eeg_total[shuffInd[:int(split_ratio*nTrials)].squeeze(), :, :]
    train_basis_set = basisSet[train_posBin]
    train_basis_set = train_basis_set# + np.random.normal(0, 0.1, train_basis_set.shape)

    test_posBin = posBin[shuffInd[int(split_ratio*nTrials):].squeeze()]  # shuffle trial order
    test_eeg_total = eeg_total[shuffInd[int(split_ratio*nTrials):].squeeze(), :, :]
    test_basis_set = basisSet[test_posBin]

    print(train_eeg_total.shape, test_eeg_total.shape)
    return train_eeg_total, train_basis_set, test_eeg_total, test_posBin, train_posBin


def plot_training_losses(training_losses):
    fig, axs = plt.subplots(3, 2)
    axs[0,0].plot(training_losses[0][-100:])
    axs[0,1].plot(training_losses[0])
    axs[1,0].plot(training_losses[1][-100:])
    axs[1,1].plot(training_losses[1])
    axs[2,0].plot(training_losses[2][-100:])
    axs[2,1].plot(training_losses[2])
    plt.show()
    
def plot_tfs(tf_total, times, save_path=None):
    tf_total = np.concatenate([tf_total, np.expand_dims(tf_total[:, :, :, -1], axis=-1)], axis=3)
    fig, ax = plt.subplots()
    tf = np.mean(np.mean(tf_total, 0), 1).transpose()
    im = ax.imshow(tf, aspect="auto", interpolation="quadric", cmap='viridis')
    plt.yticks(np.arange(9), ["$-180\degree$", "$-135\degree$", "$-90\degree$", "$-45\degree$", "$0\degree$", "$45\degree$", "$90\degree$", "$135\degree$", "$180\degree$"])
    if times.size>100:
        plt.xticks(np.where((times % 100 == 0))[0], np.where((times % 100 == 0))[0]*4)
    fig.set_size_inches(20, 3.5)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.clf()
                
    