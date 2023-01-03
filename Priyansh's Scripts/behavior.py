from scipy.io import loadmat
import numpy as np

def load_posBin(path):
    pos = loadmat(path)['ind']['cueBin'][0][0].transpose()
    for _ in range(len(pos)):
        pos[_] = pos[_]-1

    return pos.squeeze()

