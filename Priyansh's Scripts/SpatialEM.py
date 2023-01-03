import math
import numpy as np
from scipy.io import loadmat
from pytictoc import TicToc
import multiprocessing
import pickle

def tfunc():
    """ Function for the parallel processing """


def spatialEM(sn):
    """ Run spatial encoding model on evoked and total power.
    @author: Sanjana Girish """

    em = {}
    tictoc = TicToc()

    name = '_SpatialTF'  # name of files to be saved

    # setup directories
    dRoot = 'Data/'
    eRoot = 'EEG/'
    bRoot = 'Behavior/'

    # parameters to set
    nChans = 8  # No. of channels
    nBins = nChans  # No. of stimulus bins
    nIter = 10  # No. of iterations
    nBlocks = 3  # No. of blocks for cross-validation
    frequencies = np.array([[8, 12]])  # frequency bands to analyze
    bands = {'Alpha'}
    times = np.array([range(-500, 1249, 4)])  # timepoints of interest (
    # range:exclusive)

    window = 4
    Fs = 250
    nFreqs = np.shape(frequencies)[0]
    nElectrodes = 20
    nSamps = len(times[0])

    # for brevity in analysis: set up struct
    em["nChans"] = nChans
    em["nBins"] = nBins
    em["nIter"] = nIter
    em["nBlocks"] = nBlocks
    em["frequencies"] = frequencies.tolist()
    em["bands"] = frozenset(bands)
    em["time"] = times
    em["window"] = window
    em["Fs"] = Fs
    em["nElectrodes"] = nElectrodes

    # Specify basis set
    sinPower = 7
    x = np.array([np.linspace(0, 2 * math.pi - 2 * math.pi / nBins, nBins)])
    cCenters = np.array([
        np.linspace(0, 2 * math.pi - 2 * math.pi / nChans, nChans)])
    cCenters = np.rad2deg(cCenters)
    pred = np.sin(0.5 * x) ** sinPower  # hypothetical channel responses
    pred = np.roll(pred, -5)  # shift the initial basis function
    basisSet = np.empty((nChans, nBins,))
    basisSet[:] = np.nan
    for c in range(1, nChans + 1):
        basisSet[c - 1, :] = np.roll(pred,
                                     c)  # generate circularly shifted basis functions

    # ------------------------- Grab Data ------------------------------------

    # Get position bin index from behavior file
    fName = f'../{dRoot}{str(sn)}_Behavior'
    behaviorFile = loadmat(str(fName))

    posBin = behaviorFile['ind']['cueBin'][0][0].transpose()
    # Data from posBin: posBin[x, 0] where 0 <= x <= 1367
    em["posBin"] = posBin  # add to fm structure so it's saved

    # Get EEG Data
    fName = f'../{eRoot}{str(sn)}_EEG'
    eegFile = loadmat(str(fName))

    # Data from eeg_data: eeg_data[x, y, z]
    # where 0 <= x <= 1267; 0 <= y <= 21; 0 <= z <= 687
    eeg_data = eegFile['eeg']['data'][0][0]
    
    # Data from eegs: eegs[x, y, z]
    # where 0 <= x <= 1267; 0 <= y <= 19; 0 <= z <= 687
    eegs = eeg_data[:, 0:20, :]  # get scalp EEG (drop EOG electrodes)

    nt_artInd = eegFile['eeg']['arf'][0][0]['artIndCleaned'][0][0]
    artInd = nt_artInd.transpose()  # grab artifact rejection index
    # Data from artInd: artInd[x, 0] where 0 <= x <= 1367

    # Data from tois and toisRange: toisRange[0, x] where 0 <= x <= 687
    toisRange = np.array([np.arange(int(eegFile['eeg']['preTime']),
                                    int(eegFile['eeg']['postTime'] + 1), 4)])
    tois = 1 * np.isin(toisRange, times)
    nTimes = tois.size  # index time points for analysis.

    # Remove rejected trials
    not_artInd = 1 - artInd  # not_artInd[x, 0] where 0<=x<=1367
    
    # Data from eegs: eegs[x, y, x] where 0<=x<=1198 0<=y<=19 0<=z<=687
    eegs = eegs[(not_artInd.transpose()[0] == 1), :, :]

    # Data from posBin: posBin[x, 0] where 0 <= x <= 1198
    posBin = posBin[(not_artInd.transpose()[0] == 1)]
    nTrials = posBin.size  # No. of good trials
    em["nTrials"] = nTrials
    
    # ------------------------------------------------------------------------

    print("Preallocating matrices ... ")
    # Preallocate Matrices
    blocks = np.empty((nTrials, nIter))
    blocks[:] = np.nan
    em["blocks"] = blocks.tolist()  # create em.block to save block assignments
    em_blocks = np.empty((blocks.shape[0], nIter))

    tf_evoked = np.empty((nFreqs, nIter, nSamps, nBlocks, nChans))
    tf_evoked[:] = np.nan
    tf_total = tf_evoked
    C2_evoked = np.empty((nFreqs, nIter, nSamps, nBlocks, nBins, nChans))
    C2_evoked[:] = np.nan
    C2_total = C2_evoked

    # Loop through each frequency
    for f in range(1, nFreqs + 1):

        tictoc.tic()  # start timing frequency loop
        print(f"Frequency {f} out of {nFreqs}")

        # Filter Data from MATLAB
        #fName = f'../EEG_filtered/1_EEGfilt'
        fName = f'../EEG/{sn}_EEGfilt'
        eeg_filt = loadmat(str(fName))

        # Data from fdata_evoked and fdata_total: fdata[x, y, z]
        # 0 <= x <= 1198 0 <= y <= 19 0 <= z <= 687
        fdata_evoked = eeg_filt['eeg']['evoked'][0][0]
        fdata_total = eeg_filt['eeg']['total'][0][0]
        
        fdata_evoked = fdata_evoked[(not_artInd.transpose() == 1).squeeze(), 1:21, :]
        fdata_total = fdata_total[(not_artInd.transpose() == 1).squeeze(), 1:21, :]
        #fdata_evoked = eeg_filt

        # Loop through each iteration
        for iter in range(1, nIter + 1):
            print(f"Processing {iter} out of {nIter} iterations")
            # ----------------------------------------------------------
            # Assign trials to blocks (such that trials per position are
            # equated within blocks)
            # -----------------------------------------------------------

            # preallocate arrays
            blocks = np.empty((np.shape(posBin)))
            blocks[:] = np.nan
            shuffBlocks = np.empty((np.shape(posBin)))
            shuffBlocks[:] = np.nan

            # count number of trials within each position bin
            binCnt = np.empty((nBins))
            for bin in range(1, nBins + 1):
                binCnt[bin - 1] = np.sum(1 * np.equal(posBin, bin))
            binCnt = np.array([binCnt])
            # Data from binCnt: binCnt[0, x] where 0 <= x <= 7

            minCnt = min(binCnt[0])  # No. of trials for position bin with fewest trials

            # max No. of trials such that the No. of trials for each bin
            # can be equated within each block
            nPerBin = np.floor(minCnt / nBlocks)

            # shuffle trials
            shuffInd = np.array([np.random.permutation(nTrials)]).transpose()  # create shuffle index
            shuffBin = posBin[shuffInd, 0]  # shuffle trial order
            # Data from shuffInd and ShuffBin: shuff[x, 0] where 0 <= x <= 1198

            # take the 1st nPerBin x nBlocks trials for each position bin.
            for bin in range(1, nBins + 1):
                idx = np.array([np.where(shuffBin == bin)[0]]).transpose()  # get index for trials belonging to the
                # current bin ; Data from idx: idx[x, 0] where 0 <= x <= 150

                idx = np.array([idx[0: int(nPerBin * nBlocks), 0]]).transpose()  # drop excess trials
                # Data from idx: idx[x, 0] where 0 <= x <= 146

                x_nblock = np.array([[1, 2, 3]])
                x = np.tile(x_nblock, (int(nPerBin), 1))
                # Data from x: x[a, b] where 0 <= a <= 48, 0 <= b <= 2

                shuffBlocks[idx, 0] = np.array([x.ravel()]).transpose()  # assign randomly order trials to blocks

            #  unshuffle block assignment
            blocks[shuffInd, 0] = shuffBlocks
            # Data from blocks: blocks[x, 0] where 0 <= x <= 1198

            # save block assignment
            em_blocks[:, iter - 1] = blocks[:, 0]
            em["blocks"] = em_blocks  # block assignment
            nTrialsPerBlock = len(blocks[blocks == 1])  # # of trials per block

            # ----------------------------------------------------------------------------------------

            # Average data for each position bin across blocks
            posBins = np.array([range(1, nBins + 1)])
            blockDat_evoked = np.empty((nBins * nBlocks, nElectrodes, nSamps))
            blockDat_evoked[:] = np.nan  # averaged evoked data
            blockDat_total = np.empty((nBins * nBlocks, nElectrodes, nSamps))
            blockDat_total[:] = np.nan  # averaged total data
            labels = np.empty((nBins * nBlocks, 1))
            labels[:] = np.nan  # bin labels for averaged data
            blockNum = np.empty((nBins * nBlocks, 1))
            blockNum[:] = np.nan  # block numbers for averaged data
            c = np.empty((nBins * nBlocks, nChans))
            c[:] = np.nan  # predicted channel responses for averaged data
            bCnt = 1

            for ii in range(1, nBins + 1):
                for iii in range(1, nBlocks + 1):
                    blockDat_mean_cond = 1 * ((posBin == posBins[0, ii - 1]) & (blocks == iii))
                    fdata_tois = fdata_evoked[:, :, tois[0] == 1]

                    blockDat_evoked[bCnt - 1, :, :] = np.square(abs(np.squeeze(np.mean(
                        fdata_tois[blockDat_mean_cond.transpose()[0] == 1, :, :], axis=0))))
                    # Data from blockDat_evoked[x, y, z] where 0 <= x <= 23; 0 <= y <= 19; 0 <= z <= 437

                    blockDat_total[bCnt - 1, :, :] = abs(np.squeeze(np.mean(
                        fdata_tois[blockDat_mean_cond.transpose()[0] == 1, :, :], axis=0)))
                    # Data from blockDat_total[x, y, z] where 0 <= x <= 23; 0 <= y <= 19; 0 <= z <= 437

                    labels[bCnt - 1, 0] = ii
                    # Data from labels[x, 0] where 0 <= x <= 23
                    blockNum[bCnt - 1, 0] = iii
                    # Data from blockNum[x, 0] where 0 <= x <= 23
                    c[bCnt - 1, :] = basisSet[ii - 1, :]
                    # Data from c[x, y] where 0 <= x <= 23; 0 <= y <= 7
                    bCnt += 1

            for t in range(1, nSamps + 1):
                # grab data for timepoint t
                toi_array = np.array([range((times[0, t - 1] - (window // 2)), (times[0, t - 1] + (window // 2)) + 1)])
                toi = np.isin(times, toi_array)  # time window of interest
                # Data from toi: 1 x 438
                de = np.squeeze(np.mean(blockDat_evoked[:, :, toi[0, :]], axis=2))  # evoked data
                dt = np.squeeze(np.mean(blockDat_total[:, :, toi[0, :]], axis=2))  # total data
                # Data from de and dt: 24 x 20

                # Do forward model
                for i in range(1, nBlocks+1):
                    trnl = labels[blockNum != i].reshape(-1, 1)  # training labels
                    tstl = np.unique(labels[blockNum != i]).reshape(-1, 1)  # test labels

                    # -------------------------------------------------------------------------------------------------
                    #      Analysis on Evoked Power
                    # -------------------------------------------------------------------------------------------------

                    B1 = de[(blockNum != i).transpose()[0], :]  # training data
                    B2 = de[(blockNum == i).transpose()[0], :]  # test data
                    C1 = c[(blockNum != i).transpose()[0], :]  # predicted channel outputs for training data
                    W = np.linalg.lstsq(C1, B1, rcond=None)[0]  # estimate weight matrix
                    
                    
                    C2 = np.linalg.lstsq(W.conj().transpose(), B2.conj().transpose(), rcond=None)[0].conj().transpose()# estimate channel responses
                    # Data from B1: 16 x 20; B2: 8 x 20; C1: 16 x 8; W: 8 x 20;
                    
                    
                    C2_evoked[f - 1, iter - 1, t - 1, i - 1, :, :] = C2  # save the unshifted channel responses

                    # shift eegs to common center
                    n2shift = int(np.ceil(C2.shape[1] / 2))
                    for ii in range(1, C2.shape[0] + 1):
                        shiftInd = np.argmin(abs(posBins - tstl[ii - 1])[0]) + 1
                        C2[ii - 1, :] = np.roll(C2[ii - 1, :], shiftInd - n2shift - 1)

                    tf_evoked[f - 1, iter - 1, t - 1, i - 1, :] = np.mean(C2, axis=0)  # average shifted channel responses

                    # -------------------------------------------------------------------------------------------------
                    #      Analysis on Evoked Power
                    # -------------------------------------------------------------------------------------------------

                    B1 = de[(blockNum != i).transpose()[0], :]  # training data
                    B2 = de[(blockNum == i).transpose()[0], :]  # test data
                    C1 = c[(blockNum != i).transpose()[0], :]  # predicted channel outputs for training data
                    W = np.linalg.lstsq(C1, B1, rcond=None)[0]  # estimate weight matrix
                    
                    C2 = np.linalg.lstsq(W.conj().transpose(), B2.conj().transpose(), rcond=None)[0].conj().transpose()# estimate channel responses
                    # Data from B1: 16 x 20; B2: 8 x 20; C1: 16 x 8; W: 8 x 20;

                    C2_total[f - 1, iter - 1, t - 1, i - 1, :, :] = C2  # save the unshifted channel responses

                    # shift eegs to common center
                    n2shift = int(np.ceil(C2.shape[1] / 2))
                    for ii in range(1, C2.shape[0] + 1):
                        shiftInd = np.argmin(abs(posBins - tstl[ii - 1])[0]) + 1
                        C2[ii - 1, :] = np.roll(C2[ii - 1, :], shiftInd - n2shift - 1)

                    tf_total[f - 1, iter - 1, t - 1, i - 1, :] = np.mean(C2, axis=0)  # average shifted channel responses
                
        

        tictoc.toc()
    fName = f"{dRoot}{str(sn)}{name}_"
    em["C2.evoked"] = C2_evoked
    em["C2.total"] = C2_total
    em['tfs.evoked'] = tf_evoked
    em["tfs.total"] = tf_total
    em["nBlocks"] = nBlocks
    #np.savez(fName, C2_evoked=C2_evoked, C2_total=C2_total, tf_total=tf_total, tf_evoked=tf_evoked, em=em,
    #         nBlocks=nBlocks, basisSet=basisSet, blocks=blocks, nTrials=nTrials, posBin=posBin,
    #         allow_pickle=True)
    pickle.dump(em, open(f'../data/{sn}_SpatialTF_.pickle', 'wb'))
    print("Saved data to file ")


    #print('C2.evoked: ', C2_evoked.shape)
    #print('C2.total: ', C2_total.shape)
    print('tfs.evoked: ', tf_evoked.shape)
    #print('tfs.total: ', tf_total.shape)
    # To access file: data = np.load("Data/1_SpatialTF.npz", allow_pickle=True)
    # To access em dictionary, data['em'].item()
    return em

if __name__ == '__main__':
    spatialEM(1)
    spatialEM(2)
