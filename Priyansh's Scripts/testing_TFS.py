from HelperFunctions import init_TF, plot_C2
import numpy as np

tf = init_TF(8,8)
plot_C2(tf)


shiftInd = -4
for ii in range(1, tf.shape[0]+1):
    #shiftInd = np.argmin(abs(posBins - tstl[ii-1])[0])
    tf[ii-1, :] = np.roll(tf[ii-1, :], shiftInd)
    shiftInd += 1

plot_C2(tf)

