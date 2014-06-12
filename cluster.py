import numpy as np
from sklearn import mixture, cluster, metrics

fname = "blue_monkeys.npz"


d = np.load(fname)
fbanks = d['fbanks']
labels = d['labels']
strided = d['strided']
labels_strided = d['labels_strided']
stride = d['stride']
print "stride", stride
print labels_strided.shape[0]
print strided.shape
labels_set = set(labels_strided)
print labels_set

print "loaded:", fname


select = np.logical_or(labels_strided=="p", labels_strided=="h") # TODO remove
strided = strided[select]
labels_strided = labels_strided[select]

kmeans = cluster.KMeans(init='k-means++', n_clusters=2, # len(labels_set)+1
        n_init=10)
labels_pred = kmeans.fit_predict(strided)
print kmeans
print metrics.adjusted_rand_score(labels_pred, labels_strided)

dpgmm = mixture.VBGMM(n_components=2, # len(labels_set)+1
        covariance_type='diag')
dpgmm.fit(strided)
labels_pred = dpgmm.predict(strided)
print dpgmm
print metrics.adjusted_rand_score(labels_pred, labels_strided)

# TODO check / eval
# TODO spectral clustering
# TODO stack 4 seconds worth of frames (40 or 42 ;] frames) and stride them
