import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import scipy
import time
import torch
import matplotlib.pyplot as plt

from Encoder import train_encoder

from HelperFunctions import init_TF, load_eeg_and_bin, make_train_test_sets, plot_C2

np.random.seed(1000)
#torch.manual_seed(0)


def EncodingModel(sn):
    start_time = time.time()
    nChans, nBins, nElectrodes = 8,8,20

    toi_start, toi_end = 0, 704
    times = np.array([range(toi_start, toi_end, 4)]).squeeze()
    nSamps = len(times)

    basisSet = init_TF(nChans, nBins)

    eeg_path = os.path.dirname(os.getcwd())+f"\\EEG\\{sn}_EEGfilt"
    pos_bin_path = os.path.dirname(os.getcwd())+f"\\data\\{sn}_Behavior.mat"

    eeg, posBin = load_eeg_and_bin(eeg_path=eeg_path, pos_bin_path=pos_bin_path)

    nTrials = posBin.size

    tf_total = np.zeros((nSamps, nChans))
    C2_total = np.empty((nSamps, nBins, nChans))

    train_eeg_total, train_basis_set, test_eeg_total, test_posBin, train_posBin = make_train_test_sets((toi_start,toi_end), eeg, basisSet, posBin, nTrials)

    for samp in [150]:

        t = times[samp]
        #sys.stdout.write('\r                                                                   ')
        #sys.stdout.write(f"\rTime = {t} in {time.time()-start_time} seconds")
        sys.stdout.flush()
        
        B1 = train_eeg_total[:, :, samp]
        B1 = B1/B1.sum(axis=1)[:, np.newaxis]
        B2 = test_eeg_total[:, :, samp]
        B2 = B2/B2.sum(axis=1)[:, np.newaxis]
                
        C2 = train_encoder(train_eeg=B1, 
                              train_basis_set=train_basis_set, 
                              test_eeg=B2,
                              train_posBin=train_posBin,
                              test_posBin=test_posBin,
                              non_linearity=True,
                              verbose=True)
        C2 = torch.nn.Softmax()(C2).numpy()
        #C2 = C2.numpy()
        #plot_C2(C2)

        for ii in range(1, C2.shape[0]+1):
            C2[ii-1, :] = np.roll(C2[ii-1, :], test_posBin[ii-1]-5)

        tf_total[samp] = np.mean(C2, axis=0)

        #plot_C2(C2)
        
    '''fig, ax = plt.subplots()
    plt.grid(False)
    im = ax.imshow(tf_total.transpose(), aspect="auto", interpolation="quadric")
    plt.show()'''
    path = f'EncoderTFs/{sn}_EncoderTF.mat'
    scipy.io.savemat(path, dict(tfs=tf_total, times=times))

    print(np.mean(C2, axis=0))
    plt.plot(np.mean(C2, axis=0))
    plt.ylim(0,1)
    plt.show()
    

if __name__ == '__main__':
    EncodingModel(3)