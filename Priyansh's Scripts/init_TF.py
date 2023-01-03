import numpy as np

def init_TF(nChans, nBins):
    sinPower = 7
    x = np.linspace(0, 2 * np.pi - 2 * np.pi / nBins, nBins)
    cCenters = np.linspace(0, 2 * np.pi - 2 * np.pi / nChans, nChans)
    cCenters = np.rad2deg(cCenters)
    pred = np.sin(0.5 * x) ** sinPower  # hypothetical channel responses
    pred = np.roll(pred, -5)  # shift the initial basis function
    basisSet = np.empty((nChans, nBins,))
    for c in range(1, nChans + 1):
        basisSet[c - 1, :] = np.roll(pred,c) 

    return basisSet