import numpy as np
from pytictoc import TicToc
import pickle
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import scipy
import time
import torch
import matplotlib.pyplot as plt

from Perceptron import train_perceptron

from HelperFunctions import init_TF, load_eeg_and_bin, make_blocks

np.random.seed(1000)
torch.manual_seed(0)


def spatialEM(sn):
    """ Run spatial encoding model on evoked and total power.
    @author: Sanjana Girish """
    print('Running on candidate', sn)
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

    
    # Preallocate Matrices

    tf_total = np.empty((nIter, nSamps, nBlocks, nChans))
    C2_total = np.empty((nIter, nSamps, nBlocks, nBins, nChans))

    # Data from eeg_evoked and eeg_total: data[x, y, z]
    # 0 <= x <= 1198 0 <= y <= 19 0 <= z <= 687
    
    # Loop through each iteration
    
    for iter in range(nIter):
        print(f"\nProcessing {iter+1} out of {nIter} iterations at time {time.time()-start_time}")

        _, blockDat_total = make_blocks(tois=[toi_start, toi_end],
                                                        eeg=eeg,
                                                        posBin=posBin,
                                                        nBlocks=nBlocks,
                                                        nTrials=nTrials)
        # ----------------------------------------------------------------------------------------        
        for samp in [150]:
            t = times[samp]
            sys.stdout.write('\r                                                                   ')
            sys.stdout.write(f"\rTime = {t}")
            sys.stdout.flush()
            # grab data for timepoint t
            dt = np.squeeze(blockDat_total[:, :, :, (t+1000)//4]).reshape((nBlocks*nBins, nElectrodes)) # total data
            
            # Data from de and dt: 24 x 20
            # Do forward model

            training_losses, training_accs = [], []
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
                B1 = B1/B1.sum(axis=1)[:, np.newaxis]
                B2 = dt[tsti, :]  # test data
                B2 = B2/B2.sum(axis=1)[:, np.newaxis]
                C1 = c[trni, :]  # predicted channel outputs for training data
                
                model, train_loss, train_acc = train_perceptron(B1=B1, C1=C1, non_linearity=False, verbose=False)  # estimate weight matrix C1*W = B1
                training_losses.append(train_loss)
                training_accs.append(train_acc)
                C2 = model(torch.Tensor(B2).cuda()).detach().cpu().numpy()# estimate channel responses W'*C2'=B2'
                
                #W_calc = np.linalg.lstsq(B1, C1)
                #W = W_calc[0]
                
                #C2 = np.dot(B2, W)
                # Data from B1: 16 x 20; B2: 8 x 20; C1: 16 x 8; W: 8 x 20;
                C2_total[iter, samp, i, :, :] = C2  # save the unshifted channel responses
                #print(np.argmax(C2, axis=1))
                # shift eegs to common center
                shiftInd = -4
                for ii in range(1, C2.shape[0]+1):
                    #shiftInd = np.argmin(abs(posBins - tstl[ii-1])[0])
                    C2[ii-1, :] = np.roll(C2[ii-1, :], shiftInd)
                    shiftInd += 1

                #print(np.mean(C2, axis=0)/np.linalg.norm(np.mean(C2, axis=0)))
                tf_total[iter, samp, i, :] = np.mean(C2, axis=0) /np.linalg.norm(np.mean(C2, axis=0)) # average shifted channel responses
            
            fig, axs = plt.subplots(3, 2)
            axs[0,0].plot(np.array(training_accs)[0])
            axs[0,1].plot(np.array(training_losses)[0, :])
            axs[1,0].plot(np.array(training_accs)[1])
            axs[1,1].plot(np.array(training_losses)[1, ])
            axs[2,0].plot(np.array(training_accs)[2])
            axs[2,1].plot(np.array(training_losses)[2, :])
            plt.show()


            #plt.plot(np.mean(np.mean(tf_total, axis=0), axis=1)[0])
            #plt.show()

    '''fig, ax = plt.subplots()
    plt.grid(False)
    tf = np.mean(np.mean(tf_total, 0), 1).transpose()
    im = ax.imshow(tf, aspect="auto", interpolation="quadric")
    title = "Perceptron without ReLU"

    plt.title(title)
    plt.legend()
    plt.show()
    plt.clf()'''

    
    print(f"Completed at {time.time()-start_time}")
    path = f'PyLinearPerceptronTFs/{sn}_PYPerceptronTF.mat'
    scipy.io.savemat(path, dict(tfs=tf_total, times=times))

    
    

if __name__ == '__main__':
    start_time = time.time()
    v = [1,2,3,7,8,9,10,11,12,14,15,16,17,18,19,20]
    for i in v:
        spatialEM(str(i))
    
