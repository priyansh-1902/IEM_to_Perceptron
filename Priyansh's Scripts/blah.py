import numpy as np
from scipy.io import loadmat, savemat

# data_dir1 = loadmat('PyNonLinearPerceptronTFs/1_PYPerceptronTFIter1.mat')
# data_dir2 = loadmat('PyNonLinearPerceptronTFs/1_PYPerceptronTFIter2.mat')
# data_dir3 = loadmat('PyNonLinearPerceptronTFs/1_PYPerceptronTFIter3.mat')
# data_dir4 = loadmat('PyNonLinearPerceptronTFs/1_PYPerceptronTFIter4.mat')
# data_dir5 = loadmat('PyNonLinearPerceptronTFs/1_PYPerceptronTFIter5.mat')


# tf_total = np.concatenate([data_dir1['tfs'], data_dir2['tfs'], data_dir3['tfs'], data_dir4['tfs'], data_dir5['tfs']], axis=0)

# C2 = np.concatenate([data_dir1['C2'], data_dir2['C2'], data_dir3['C2'], data_dir4['C2'], data_dir5['C2']], axis=0)

# path = f'PyNonLinearPerceptronTFs/1_PYPerceptronTF.mat'
# savemat(path, dict(tfs=tf_total, C2=C2, times=data_dir1['times']))




data_dir1 = loadmat('PyNonLinearPerceptronTFs/1_PYPerceptronTF.mat')
data_dir2 = loadmat('PyNonLinearPerceptronTFs/1_PYPerceptronTFIter1-5_0_200.mat')

print(data_dir1['tfs'].shape)
print(data_dir2['tfs'].shape)

tf_total = np.concatenate([data_dir2['tfs'][:, :50, :, :], data_dir1['tfs']], axis=1)
print(tf_total.shape)

C2_total = np.concatenate([data_dir2['C2'][:, :50, :, :, :], data_dir1['C2']], axis=1)
print(C2_total.shape)


print(data_dir1['times'].shape)

print(data_dir2['times'].shape)
times = np.concatenate([data_dir2['times'][:, :50], data_dir1['times']], axis=1)
print(times.shape)

path = f'PyNonLinearPerceptronTFs/1_PYPerceptronTF.mat'
savemat(path, dict(tfs=tf_total, C2=C2_total, times=times))