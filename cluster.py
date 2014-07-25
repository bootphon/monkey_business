import numpy as np
from sklearn import mixture, cluster, metrics

from mielke_replication import load_all_monkeys, combine_labels
MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus']

X, y, labelset = load_all_monkeys()
print labelset

clusterizers = [{
    'Titi_monkeys': cluster.KMeans(init='k-means++', n_clusters=
        len(labelset['Titi_monkeys']) + 1, n_init=10),
    'Blue_monkeys': cluster.KMeans(init='k-means++', n_clusters=
        len(labelset['Blue_monkeys']) + 1, n_init=10),
    'colobus': cluster.KMeans(init='k-means++', n_clusters=
        len(labelset['colobus']) + 1, n_init=10),
    'combined': cluster.KMeans(init='k-means++', n_clusters=7, n_init=10)}, {
    'Titi_monkeys': mixture.VBGMM(n_components=
        len(labelset['Titi_monkeys']) + 1, covariance_type='diag'),
    'Blue_monkeys': mixture.VBGMM(n_components=
        len(labelset['Blue_monkeys']) + 1, covariance_type='diag'),
    'colobus': mixture.VBGMM(n_components=
        len(labelset['colobus']) + 1, covariance_type='diag'),
    'combined': mixture.VBGMM(n_components=7, covariance_type='diag')}]

for clusterizer in clusterizers:
    # 1. classify calls per monkey
    for monkey in MONKEYS:
        print monkey
        clusterizer[monkey].fit(X[monkey])
        print X[monkey].shape
        y_pred = clusterizer[monkey].predict(X[monkey])
        print metrics.adjusted_rand_score(y_pred, y[monkey])

    # 2. classify over all monkeys and calls
    X_comb, y_comb, labelset_comb = combine_labels(X, y, labelset)
    clusterizer['combined'].fit(X_comb)
    y_pred = clusterizer['combined'].predict(X_comb)
    print 'COMBINED:'
    print metrics.adjusted_rand_score(y_pred, y_comb)



# TODO check / eval
# TODO spectral clustering
# TODO stack 4 seconds worth of frames (40 or 42 ;] frames) and stride them
