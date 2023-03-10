from HelperFunctions import mexican_hat, init_mexican_hat_tf


from scipy.io import loadmat, savemat

data_dir1 = loadmat('./PyNonLinearPerceptronTFs/3_PYPerceptronTF1')
data_dir2 = loadmat('./PyNonLinearPerceptronTFs/3_PYPerceptronTF2')


print(data_dir1['tfs'].shape)

print(data_dir2['tfs'].shape)

import numpy as np

tf = np.concatenate((data_dir1['tfs'], data_dir2['tfs']), axis=1)

path = f'PyNonLinearPerceptronTFs/3_PYPerceptronTF.mat'

toi_start = 0
toi_end = 177*4
times = np.array([range(toi_start, toi_end, 4)]).squeeze()
    
savemat(path, dict(tfs=tf, times=times))
