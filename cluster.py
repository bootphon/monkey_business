import numpy as np
from sklearn import mixture

fname = "blue_monkeys.npz"

d = np.load(fname)
fbanks = d['fbanks']
labels = d['labels']

dpgmm = mixture.DPGMM(n_components=10, covariance_type='diag')
print "loaded:", fname
dpgmm.fit(fbanks)
print dpgmm

# TODO check / eval
# TODO spectral clustering
# TODO stack 4 seconds worth of frames (40 or 42 ;] frames) and stride them
