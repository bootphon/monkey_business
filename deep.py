import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from nnet import DropoutNet, RegularizedNet, add_fit_score_predict
from nnet import ReLU, LogisticRegression

from mielke_replication import load_all_monkeys, combine_labels, make_monkey_set
MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus']
DROPOUT = False

def train_deep():
    add_fit_score_predict(DropoutNet)
    add_fit_score_predict(RegularizedNet)

    X, y, labelset = load_all_monkeys()
    # 1. classify calls per monkey
    for monkey in MONKEYS:
        print monkey
        X_train, X_test, y_train, y_test = train_test_split(X[monkey],
                                                            y[monkey],
                                                            test_size=0.2)
        X_train = np.asarray(X_train, dtype='float32')
        X_test = np.asarray(X_test, dtype='float32')
        y_train = np.asarray(y_train, dtype='int32')
        y_test = np.asarray(y_test, dtype='int32')
        numpy_rng = np.random.RandomState(123)
        if DROPOUT:
            clf = DropoutNet(numpy_rng=numpy_rng,
                    n_ins=X_train.shape[1],
                    layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
                    layers_sizes=[2000, 2000, 2000, 2000],
                    dropout_rates=[0., 0.5, 0.5, 0.5, 0.5],
                    n_outs=len(labelset[monkey]),
                    fast_drop=False,
                    debugprint=0)
        else:
            clf = RegularizedNet(numpy_rng=numpy_rng,
                    n_ins=X_train.shape[1],
                    layers_types=[ReLU, ReLU, LogisticRegression],
                    layers_sizes=[200, 200], 
                    n_outs=len(labelset[monkey]),
                    debugprint=0)
        clf.fit(X_train, y_train, max_epochs=200, verbose=True)
        #clf.fit(x_train, y_train, max_epochs=n_epochs, method=method, verbose=VERBOSE, plot=PLOT)
        y_pred = clf.predict(X_test)
        print metrics.classification_report(y_test, y_pred,
                                            target_names=labelset[monkey])

    # 2. classify over all monkeys and calls
    X_comb, y_comb, labelset_comb = combine_labels(X, y, labelset)
    X_train, X_test, y_train, y_test = train_test_split(X_comb, y_comb,
                                                        test_size=0.2)

    X_train = np.asarray(X_train, dtype='float32')
    X_test = np.asarray(X_test, dtype='float32')
    y_train = np.asarray(y_train, dtype='int32')
    y_test = np.asarray(y_test, dtype='int32')
    if DROPOUT:
        clf = DropoutNet(numpy_rng=numpy_rng,
                n_ins=X_train.shape[1],
                layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
                layers_sizes=[2000, 2000, 2000, 2000],
                dropout_rates=[0., 0.5, 0.5, 0.5, 0.5],
                n_outs=len(labelset_comb),
                fast_drop=False,
                debugprint=0)
    else:
        clf = RegularizedNet(numpy_rng=numpy_rng,
                n_ins=X_train.shape[1],
                layers_types=[ReLU, ReLU, LogisticRegression],
                layers_sizes=[200, 200], 
                n_outs=len(labelset_comb),
                debugprint=0)
    clf.fit(X_train, y_train, max_epochs=200, verbose=True)
    y_pred = clf.predict(X_test)
    print 'COMBINED:'
    print metrics.classification_report(y_test, y_pred,
                                        target_names=labelset_comb)


if __name__ == '__main__':
    train_deep()

