from scipy.io import loadmat, savemat
import numpy as np

data_dir1 = loadmat('PyNonLinearPerceptronTFs/3_PyPerceptronTF_Iter1-5.mat')
tfs1 = data_dir1['tfs']
times1 = data_dir1['times']

print(tfs1.shape, times1.shape)

data_dir2 = loadmat('PyNonLinearPerceptronTFs/3_PyPerceptronTF_Iter6-10.mat')
tfs2 = data_dir2['tfs']
times2 = data_dir2['times']

print(tfs2.shape, times2.shape)


tfs = np.concatenate([tfs1, tfs2], axis=0)
times = times1

print(tfs.shape, times.shape)

path = f'PyNonLinearPerceptronTFs/3_PYPerceptronTF.mat'
    
savemat(path, dict(tfs=tfs, times=times))