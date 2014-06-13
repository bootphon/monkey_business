import numpy as np
from sklearn import mixture, cluster, metrics
from extract_data import load_from_npz

fname = "blue_monkeys.npz"

fbanks, labels, strided, labels_strided, stride, labels_set = load_from_npz(
        fname)


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
