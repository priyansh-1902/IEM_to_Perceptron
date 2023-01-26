import numpy as np
from pytictoc import TicToc
import pickle
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import scipy

from matplotlib import pyplot as plt
from PIL import  ImageGrab

from HelperFunctions import init_TF, load_eeg_and_bin, make_blocks, EEG


def spatialEM(sn):
    """ Run spatial encoding model on evoked and total power.
    @author: Sanjana Girish """

    # parameters to set
    nChans = 8  # No. of channels
    nBins = nChans  # No. of stimulus bins
    nIter = 10  # No. of iterations
    nBlocks = 3  # No. of blocks for cross-validation
    frequencies = np.array([[8, 12]])  # frequency bands to analyze
    bands = {'Alpha'}
    times = np.array([range(-500, 1249, 4)]).squeeze()  # timepoints of interest (range:exclusive)

    window = 4
    Fs = 250
    nFreqs = np.shape(frequencies)[0]
    nElectrodes = 20
    nSamps = len(times)

    toi_start = -500
    toi_end = 1249
    
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

    """fig, axs = plt.subplots(20)
    for i in range(len(axs)):
        axs[i].plot(eeg.eeg_total()[0, i])
    
    fig.show()"""
    # Loop through each iteration
    for iter in range(nIter):
        print(f"Processing {iter+1} out of {nIter} iterations")

        blockDat_evoked, blockDat_total = make_blocks(tois=[toi_start, toi_end],
                                                        eeg=eeg,
                                                        posBin=posBin,
                                                        nBlocks=nBlocks,
                                                        nTrials=nTrials)
        # ----------------------------------------------------------------------------------------
        posBins = np.array([range(1, nBins + 1)])
        
        for samp in range(len(times)):
            t = times[samp]
            
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

                W_calculation = np.linalg.lstsq(C1, B1, rcond=None)  # estimate weight matrix C1*W = B1
                
                W = W_calculation[0]

                C2_calculation = np.linalg.lstsq(W.transpose(), B2.transpose(), rcond=None)# estimate channel responses W'*C2'=B2'
            
                C2 = C2_calculation[0].transpose()

                # Data from B1: 16 x 20; B2: 8 x 20; C1: 16 x 8; W: 8 x 20;
                C2_total[iter, samp, i, :, :] = C2  # save the unshifted channel responses

                # shift eegs to common center
                n2shift = int(np.ceil(C2.shape[1] / 2))
                for ii in range(C2.shape[0]):
                    shiftInd = np.argmin(abs(posBins - tstl[ii])[0])
                    C2[ii, :] = np.roll(C2[ii, :], shiftInd - n2shift)
                
                tf_total[iter, samp, i, :] = np.mean(C2, axis=0)  # average shifted channel responses


    print(np.mean(np.mean(tf_total, 0), 1).shape)
    fig, ax = plt.subplots()
    plt.grid(False)
    tf = np.mean(np.mean(tf_total, 0), 1).transpose()
    im = ax.imshow(tf, aspect="auto", interpolation="quadric")
    plt.title("Priyansh's script")
    plt.legend()
    plt.show()
    plt.clf()
    #scipy.io.savemat('1_PYSpatialTF.mat', dict(tfs=tf_total, times=times))

    
    

if __name__ == '__main__':
    spatialEM(1)
    
