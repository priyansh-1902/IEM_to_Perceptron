import numpy as np
import sys
import os
import warnings
import scipy
import time
import torch
import multiprocessing

from PerceptronEM import f, save_data

from HelperFunctions import init_TF, load_eeg_and_bin, make_blocks

warnings.filterwarnings("ignore")
# np.random.seed(1000)
# torch.manual_seed(0)


 


if __name__=='__main__':
    start_time = 0
    end_time = 600
    nIter = 2
    
    sn = sys.argv[1]
    path = f'PyNonLinearPerceptronTFs/{sn}_PYPerceptronTFIter1-2.mat'

    
    nChans, nBins, nElectrodes = 8, 8, 20  # No. of channels
    nBlocks = 3  # No. of blocks for cross-validation
    
    times = np.arange(start_time, end_time+4, 4) # timepoints of interest (range:exclusive)
    nSamps = len(times)

    basisSet = init_TF(nChans, nBins)

    eeg_path = os.path.dirname(os.getcwd())+f"\\EEG\\{sn}_EEGfilt"
    pos_bin_path = os.path.dirname(os.getcwd())+f"\\data\\{sn}_Behavior.mat"

    eeg, posBin = load_eeg_and_bin(eeg_path, pos_bin_path)

    nTrials = posBin.size

    tf_total = np.empty((nIter, nSamps, nBlocks, nChans))
    C2_total = np.empty((nIter, nSamps, nBlocks, nBins, nChans))

    trnl = np.tile(np.array(range(nBins)), nBlocks-1)  # training labels
    tstl = np.array(range(nBins))  # test labels
    c = basisSet[np.concatenate((trnl, tstl)), :]

    procs = []
    queue = multiprocessing.Queue()
    for iteration in range(nIter):
        print(f"\nProcessing {iteration+1} out of {nIter} iterations")

        blockDat_total = make_blocks(tois=[start_time, end_time],
                                                        eeg=eeg,
                                                        posBin=posBin,
                                                        nBlocks=nBlocks,
                                                        nTrials=nTrials)
        
        for samp in range(len(times)-1, -1, -1):

            t = times[samp]

            # grab data for timepoint t, dt: 24 x 20
            dt = np.squeeze(blockDat_total[:, :, :, (t+1000)//4]).reshape((nBlocks*nBins, nElectrodes))

            training_losses = []
            
            for i in range(nBlocks):
    
                trni, tsti = (np.array(range(nBlocks*nBins))//nBins)!=i, (np.array(range(nBlocks*nBins))//nBins)==i
                # -------------------------------------------------------------------------------------------------
                #      Analysis on Total Power
                # -------------------------------------------------------------------------------------------------
                
                B1, B2 = dt[trni, :], dt[tsti, :]  # training data, test data
                B1 = B1/B1.sum(axis=1)[:, np.newaxis]
                B2 = B2/B2.sum(axis=1)[:, np.newaxis]
                C1 = c[trni, :]  # predicted channel outputs for training data

                procs.append(multiprocessing.Process(target=f, args=(B1, C1, B2, C2_total, tf_total, iteration, samp, i, t, queue)))

    print(f'Starting processes. Total to run = {len(procs)}')
    while(len(procs)>0):
        running = []
        print(min(16, len(procs)))
        for i in range(min(16, len(procs))):
            running.append(procs.pop())
            running[-1].start()

        for i in range(len(running)):
            running[i].join()

        print(queue.qsize())

        while(queue.qsize()>0):
            
            C2, iteration, samp, i = queue.get()
            C2_total[iteration, samp, i, :, :] = C2  # save the unshifted channel responses
    
            # shift eegs to common center
            shiftInd = -4
            for ii in range(1, C2.shape[0]+1):
                C2[ii-1, :] = np.roll(C2[ii-1, :], shiftInd)
                shiftInd += 1

            tf_total[iteration, samp, i, :] = np.mean(C2, axis=0) /np.linalg.norm(np.mean(C2, axis=0)) # average shifted channel responses


    
    scipy.io.savemat(path, dict(tfs=tf_total, C2=C2_total, times=times))
 



