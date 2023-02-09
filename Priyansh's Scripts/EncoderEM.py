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

from HelperFunctions import init_TF, load_eeg_and_bin, make_blocks

np.random.seed(1000)
torch.manual_seed(0)


def spatialEM(sn, non_linearity=False):
    """ Run spatial encoding model on evoked and total power.
    @author: Sanjana Girish """
    if non_linearity:
        print('Running Perceptron EM with ReLU')
    else:
        print('Running Perceptron EM without non-linearity')
    # parameters to set
    nChans = 8  # No. of channels
    nBins = nChans  # No. of stimulus bins
    
    nBlocks = 3  # No. of blocks for cross-validation
    frequencies = np.array([8, 12])  # frequency bands to analyze
    bands = {'Alpha'}
    

    window = 4
    Fs = 250
    nFreqs = np.shape(frequencies)[0]
    nElectrodes = 20
    nIter = 10

    toi_start = 0
    toi_end = 704
    times = np.array([range(toi_start, toi_end, 4)]).squeeze()  # timepoints of interest (range:exclusive)
    nSamps = len(times)

    # Specify basis set
    basisSet = init_TF(nChans, nBins)

    # ------------------------- Grab Data ------------------------------------
    eeg_path = os.path.dirname(os.getcwd())+f"\\EEG\\{sn}_EEGfilt"
    pos_bin_path = os.path.dirname(os.getcwd())+f"\\data\\{sn}_Behavior.mat"

    # Data from posBin: posBin[x, 0] where 0 <= x <= 1367
    # Data from eegs: eegs[x, y, x] where 0<=x<=1198 0<=y<=19 0<=z<=687
    eeg, posBin = load_eeg_and_bin(eeg_path, pos_bin_path)

    # Data from tois and toisRange: toisRange[0, x] where 0 <= x <= 687
    toisRange = np.arange(int(eeg.preTime), int(eeg.postTime + 1), 4)
    tois = 1 * np.isin(toisRange, times)
    nTimes = tois.size  # index time points for analysis.

    nTrials = posBin.size  # No. of good trials
    
    # ------------------------------------------------------------------------

    print("Preallocating matrices ... ")
    # Preallocate Matrices

    tf_total = np.empty((nIter, nSamps, nBlocks, nChans))
    C2_total = np.empty((nIter, nSamps, nBlocks, nBins, nChans))

    # Data from eeg_evoked and eeg_total: data[x, y, z]
    # 0 <= x <= 1198 0 <= y <= 19 0 <= z <= 687
    
    # Loop through each iteration
    start_time = time.time()
    for iter in range(nIter):
        print(f"\nProcessing {iter+1} out of {nIter} iterations at time {time.time()-start_time}")

        _, blockDat_total = make_blocks(tois=[toi_start, toi_end],
                                                        eeg=eeg,
                                                        posBin=posBin,
                                                        nBlocks=nBlocks,
                                                        nTrials=nTrials)
        # ----------------------------------------------------------------------------------------        
        for samp in range(len(times)):
            t = times[samp]
            sys.stdout.write('\r                                                                   ')
            sys.stdout.write(f"\rTime = {t}")
            sys.stdout.flush()
            # grab data for timepoint t
            dt = np.squeeze(blockDat_total[:, :, :, (t+1000)//4]).reshape((nBlocks*nBins, nElectrodes)) # total data
            
            # Data from de and dt: 24 x 20
            # Do forward model
            for i in range(nBlocks):
                trnl = np.tile(np.array(range(nBins)), nBlocks-1)  # training labels
                tstl = np.array(range(nBins))  # test labels
                c = basisSet[np.concatenate((trnl, tstl)), :]
                trni = (np.array(range(nBlocks*nBins))//nBins)!=i
                tsti = (np.array(range(nBlocks*nBins))//nBins)==i
                # -------------------------------------------------------------------------------------------------
                #      Analysis on Total Power
                # -------------------------------------------------------------------------------------------------
                
                B1 = dt[trni, :]  # training data
                B2 = dt[tsti, :]  # test data
                C1 = c[trni, :]  # predicted channel outputs for training data
                
                W_calculation = train_encoder(B1=B1, C1=C1, non_linearity=non_linearity, verbose=False)  # estimate weight matrix C1*W = B1
                
                W = W_calculation
                print(B1)
                return
                C2_calculation = np.linalg.lstsq(W.transpose(), B2.transpose(), rcond=None)# estimate channel responses W'*C2'=B2'
            
                C2 = C2_calculation[0].transpose()
                
                # Data from B1: 16 x 20; B2: 8 x 20; C1: 16 x 8; W: 8 x 20;
                C2_total[iter, samp, i, :, :] = C2  # save the unshifted channel responses
                
                # shift eegs to common center
                shiftInd = -4
                for ii in range(1, C2.shape[0]+1):
                    #shiftInd = np.argmin(abs(posBins - tstl[ii-1])[0])
                    C2[ii-1, :] = np.roll(C2[ii-1, :], shiftInd)
                    shiftInd += 1

                tf_total[iter, samp, i, :] = np.mean(C2, axis=0)  # average shifted channel responses
                

    fig, ax = plt.subplots()
    plt.grid(False)
    tf = np.mean(np.mean(tf_total, 0), 1).transpose()
    im = ax.imshow(tf, aspect="auto", interpolation="quadric")
    if non_linearity:
        title = "Perceptron with ReLU"
    else:
        title = "Perceptron without ReLU"

    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()

    if non_linearity:
        path = f'PyNonLinearPerceptronTFs/{sn}_PYPerceptronTF.mat'
    else:
        path = f'PyLinearPerceptronTFs/{sn}_PYPerceptronTF.mat'
    scipy.io.savemat(path, dict(tfs=tf_total, times=times))

    
    

if __name__ == '__main__':
    v = [1,2,3,7,8,9,10,11,12,14,15,16,17,18,19,20]
    for i in [3]:
        #spatialEM(str(i), non_linearity=False)
        spatialEM(str(i), non_linearity=True)
    
