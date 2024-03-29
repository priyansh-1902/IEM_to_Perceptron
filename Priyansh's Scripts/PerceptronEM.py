import numpy as np
import sys
import os
import warnings
import scipy
import time
from datetime import datetime
import torch
import matplotlib.pyplot as plt

from Perceptron import LinearPerceptron, NonLinearMLP, NonLinearPerceptron, train_perceptron_cpu

import HelperFunctions as hf

warnings.filterwarnings("ignore")
np.random.seed(1000)
torch.manual_seed(0)

def save_data(sn, tf_total, C2_total, times):
    path = f'{sn}_TF.mat'
    scipy.io.savemat(path, dict(tfs=tf_total, C2=C2_total, times=times))


def trainEM(model, B1, C1, B2, C2_total, tf_total, iteration, samp, i, t, verbose, queue=None):
    model, train_loss, epoch = train_perceptron_cpu(model, B1=B1, C1=C1, verbose=False)  # estimate weight matrix C1*W = B1

    C2 = model(torch.Tensor(B2)).detach().numpy()# estimate channel responses W'*C2'=B2'

    # Data from B1: 16 x 20; B2: 8 x 20; C1: 16 x 8; W: 8 x 20;
    
    C2_total[iteration, samp, i, :, :] = C2  # save the unshifted channel responses
    
    # shift eegs to common center
    shiftInd = -4
    for ii in range(1, C2.shape[0]+1):
        C2[ii-1, :] = np.roll(C2[ii-1, :], shiftInd)
        shiftInd += 1

    tf_total[iteration, samp, i, :] = np.mean(C2, axis=0) /np.linalg.norm(np.mean(C2, axis=0)) # average shifted channel responses
    
    if verbose==1:
        print(f"{datetime.now().strftime('%H:%M:%S')}: Trained {t} for {epoch} epochs upto a loss of {round(train_loss[-1], 5)}")
 
    


def PerceptronEM(sn, model, nIter, start_time, end_time, startTime, verbose, save_heat_map):
    print('Running on candidate', sn)

    if model == 'LinearPerceptron':
        _model = LinearPerceptron
    elif model == 'NonLinearPerceptron':
        _model = NonLinearPerceptron
    else:
        _model = NonLinearMLP

    nChans, nBins, nElectrodes = 8, 8, 20  # No. of channels
    nBlocks = 3  # No. of blocks for cross-validation

    times = np.arange(start_time, end_time+4, 4) # timepoints of interest (range:exclusive)
    nSamps = len(times)

    # Specify basis set
    basisSet = hf.init_TF(nChans, nBins)

    eeg_path = os.path.dirname(os.getcwd())+f"/EEG/{sn}_EEGfilt"
    pos_bin_path = os.path.dirname(os.getcwd())+f"/data/{sn}_Behavior.mat"

    # Data from posBin: posBin[x, 0] where 0 <= x <= 1367
    # Data from eegs: eegs[x, y, x] where 0<=x<=1198 0<=y<=19 0<=z<=687
    eeg, posBin = hf.load_eeg_and_bin(eeg_path, pos_bin_path)

    nTrials = posBin.size  # No. of good trials

    # ------------------------------------------------------------------------

    # Preallocate Matrices
    tf_total = np.empty((nIter, nSamps, nBlocks, nChans))
    C2_total = np.empty((nIter, nSamps, nBlocks, nBins, nChans))

    # Data from eeg_evoked and eeg_total: data[x, y, z]
    # 0 <= x <= 1198 0 <= y <= 19 0 <= z <= 687

    # Loop through each iteration
    for iteration in range(nIter):
        print(f"\nProcessing {iteration+1} out of {nIter} iterations at time {datetime.now().strftime('%H:%M:%S')}")

        blockDat_total = hf.make_blocks(tois=[start_time, end_time],
                                                        eeg=eeg,
                                                        posBin=posBin,
                                                        nBlocks=nBlocks,
                                                        nTrials=nTrials)
        # ----------------------------------------------------------------------------------------        


        for samp in range(len(times)):
            epochs = []
            losses = []
            t = times[samp]

            # grab data for timepoint t, dt: 24 x 20
            dt = np.squeeze(blockDat_total[:, :, :, (t+1000)//4]).reshape((nBlocks*nBins, nElectrodes))

            for i in range(nBlocks):
                trnl = np.tile(np.array(range(nBins)), nBlocks-1)  # training labels
                tstl = np.array(range(nBins))  # test labels
                c = basisSet[np.concatenate((trnl, tstl)), :]
                trni, tsti = (np.array(range(nBlocks*nBins))//nBins)!=i, (np.array(range(nBlocks*nBins))//nBins)==i
                # -------------------------------------------------------------------------------------------------
                #      Analysis on Total Power
                # -------------------------------------------------------------------------------------------------
                
                B1, B2 = dt[trni, :], dt[tsti, :]  # training data, test data
                B1 = B1/B1.sum(axis=1)[:, np.newaxis]
                B2 = B2/B2.sum(axis=1)[:, np.newaxis]
                C1 = c[trni, :]  # predicted channel outputs for training data

                trainEM(_model, B1, C1, B2, C2_total, tf_total, iteration, samp, i, t, verbose=verbose)

            
            # plot_training_losses(training_losses)

    hf.plot_tfs(tf_total, times, save_path=save_heat_map)

    save_data(sn, tf_total=tf_total, C2_total=C2_total, times=times)
    print(f"Completed at {time.time()-startTime}")


if __name__ == '__main__':
    startTime=time.time()
    PerceptronEM(sn=sys.argv[1], nIter=10, startTime=time.time(), start_time=0, end_time=600)

