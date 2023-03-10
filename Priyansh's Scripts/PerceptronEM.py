import numpy as np
import sys
import os
import warnings
import scipy
import time
import torch
import matplotlib.pyplot as plt

from Perceptron import train_perceptron_cpu, NonLinearPerceptron, LinearPerceptron

from HelperFunctions import init_TF, load_eeg_and_bin, make_blocks, plot_training_losses

warnings.filterwarnings("ignore")
# np.random.seed(1000)
# torch.manual_seed(0)

def save_data(sn, tf_total, C2_total, times):
    path = f'PyNonLinearPerceptronTFs/{sn}_PYPerceptronTFIter{sys.argv[2]}.mat'
    scipy.io.savemat(path, dict(tfs=tf_total, C2=C2_total, times=times))


def PerceptronEM(sn, nIter, start_time, end_time):
    print('Running on candidate', sn)
    startTime = time.time()

    nChans, nBins, nElectrodes = 8, 8, 20  # No. of channels
    nBlocks = 3  # No. of blocks for cross-validation

    times = np.arange(start_time, end_time, 4) # timepoints of interest (range:exclusive)
    nSamps = len(times)

    # Specify basis set
    basisSet = init_TF(nChans, nBins)

    eeg_path = os.path.dirname(os.getcwd())+f"\\EEG\\{sn}_EEGfilt"
    pos_bin_path = os.path.dirname(os.getcwd())+f"\\data\\{sn}_Behavior.mat"

    # Data from posBin: posBin[x, 0] where 0 <= x <= 1367
    # Data from eegs: eegs[x, y, x] where 0<=x<=1198 0<=y<=19 0<=z<=687
    eeg, posBin = load_eeg_and_bin(eeg_path, pos_bin_path)

    nTrials = posBin.size  # No. of good trials

    # ------------------------------------------------------------------------

    # Preallocate Matrices
    tf_total = np.empty((nIter, nSamps, nBlocks, nChans))
    C2_total = np.empty((nIter, nSamps, nBlocks, nBins, nChans))

    # Data from eeg_evoked and eeg_total: data[x, y, z]
    # 0 <= x <= 1198 0 <= y <= 19 0 <= z <= 687

    # Loop through each iteration
    for iteration in range(nIter):
        print(f"\nProcessing {iteration+1} out of {nIter} iterations at time {time.time()-startTime}")

        blockDat_total = make_blocks(tois=[start_time, end_time],
                                                        eeg=eeg,
                                                        posBin=posBin,
                                                        nBlocks=nBlocks,
                                                        nTrials=nTrials)
        # ----------------------------------------------------------------------------------------        
        
        non_linearity = True
        if (non_linearity):
            models = [NonLinearPerceptron() for i in range(3)]
        else:
            model = LinearPerceptron().cuda()

        for samp in [90]:
            epochs = []
            losses = []
            t = times[samp]

            # grab data for timepoint t, dt: 24 x 20
            dt = np.squeeze(blockDat_total[:, :, :, (t+1000)//4]).reshape((nBlocks*nBins, nElectrodes))

            training_losses = []
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

                models[i], train_loss, epoch = train_perceptron_cpu(model=models[i], B1=B1, C1=C1, verbose=False)  # estimate weight matrix C1*W = B1
                
                losses.append(str(round(train_loss[-1], 3)))
                epochs.append(epoch)
                training_losses.append(train_loss)
                
                C2 = models[i](torch.Tensor(B2)).detach().numpy()# estimate channel responses W'*C2'=B2'

                # Data from B1: 16 x 20; B2: 8 x 20; C1: 16 x 8; W: 8 x 20;
                C2_total[iteration, samp, i, :, :] = C2  # save the unshifted channel responses
                
                # shift eegs to common center
                shiftInd = -4
                for ii in range(1, C2.shape[0]+1):
                    C2[ii-1, :] = np.roll(C2[ii-1, :], shiftInd)
                    shiftInd += 1

                tf_total[iteration, samp, i, :] = np.mean(C2, axis=0) /np.linalg.norm(np.mean(C2, axis=0)) # average shifted channel responses
            
            print(f"\rTrained {t} at {time.time()-startTime} seconds for {epochs} epochs and to a loss of {losses}")
 

            plot_training_losses(training_losses)


    # save_data()
    print(f"Completed at {time.time()-startTime}")


if __name__ == '__main__':
    PerceptronEM(sn=sys.argv[1], nIter=1, start_time=200, end_time=600)

