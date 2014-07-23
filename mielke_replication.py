#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: mielke_replication.py
# date: Mon July 21 18:13 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""mielke_replication: replication of some of the results described in
Mielke, Zuberbuehler (2013) A method for automated individual, species and
call type recognition in free-ranging animals, Animal Behaviour 86, pp 475--482

"""

from __future__ import division

import os.path as path
import cPickle as pickle

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as metrics

import numpy as np

MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus']

from corpus import BASEDIR, load_data_stacked

def load_all_monkeys():
    """Load stacked data for all monkeys.

    :return
      X: dict from monkeyname to ndarray containing the stacked audio
      y: dict from monkeyname to ndarray containing the labels (as ints)
      labelset: dict from monkeyname to sorted list of call names
    """
    _memo_fname = path.join(BASEDIR, 'monkey_calls_stacked.pkl')
    if path.exists(_memo_fname):
        with open(_memo_fname, 'rb') as fid:
            X, y, labelset = pickle.load(fid)
    else:
        labelset = {}
        X = {}
        y = {}
        for monkey in MONKEYS:
            X_, y_, labelset_ = load_data_stacked(monkey)
            labelset[monkey] = labelset_
            X[monkey] = X_
            y[monkey] = y_

        with open('monkey_data.pkl', 'wb') as fid:
            pickle.dump((X, y, labelset), fid, -1)
    return X, y, labelset


def combine_labels(X, y, labelset):
    combined_labels = []
    for monkey in labelset:
        for label in labelset[monkey]:
            combined_labels.append(monkey + '-' + label)
    combined_labels = sorted(combined_labels)
    comblabel2idx = dict(zip(combined_labels, range(len(combined_labels))))

    X_comb = None
    y_comb = None
    for comb_label in combined_labels:
        monkey, label = comb_label.split('-')
        idx2label = dict(zip(range(len(labelset[monkey])), labelset[monkey]))
        X_ = X[monkey]
        y_ = np.array([comblabel2idx[monkey + '-' + idx2label[x]]
                       for x in y[monkey]])

        if X_comb is None:
            X_comb = X_
            y_comb = y_
        else:
            X_comb = np.vstack((X_comb, X_))
            y_comb = np.hstack((y_comb, y_))

    return X_comb, y_comb, combined_labels


def make_monkey_set(X, y, labelset):
    monkeys = sorted(labelset.keys())
    label2idx = dict(zip(monkeys, range(len(monkeys))))
    X_comb = None
    y_comb = None
    for monkey in monkeys:
        X_ = X[monkey]
        y_ = np.ones(X_.shape[0]) * label2idx[monkey]
        if X_comb is None:
            X_comb = X_
            y_comb = y_
        else:
            X_comb = np.vstack((X_comb, X_))
            y_comb = np.hstack((y_comb, y_))
    return X_comb, y_comb, monkeys


def replicate():
    X, y, labelset = load_all_monkeys()
    # 1. classify calls per monkey
    for monkey in MONKEYS:
        print monkey
        X_train, X_test, y_train, y_test = train_test_split(X[monkey],
                                                            y[monkey],
                                                            test_size=0.2)
        clf = SVC(kernel='rbf', C=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print metrics.classification_report(y_test, y_pred,
                                            target_names=labelset[monkey])

    # 2. classify over all monkeys and calls
    X_comb, y_comb, labelset_comb = combine_labels(X, y, labelset)
    X_train, X_test, y_train, y_test = train_test_split(X_comb, y_comb,
                                                        test_size=0.2)
    clf = SVC(kernel='rbf', C=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print 'COMBINED:'
    print metrics.classification_report(y_test, y_pred,
                                        target_names=labelset_comb)

    # 3. classify monkeys
    X_comb, y_comb, labelset_comb = make_monkey_set(X, y, labelset)
    X_train, X_test, y_train, y_test = train_test_split(X_comb, y_comb,
                                                        test_size=0.2)
    clf = SVC(kernel='rbf', C=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print 'SPECIES:'
    print metrics.classification_report(y_test, y_pred,
                                        target_names=labelset_comb)


if __name__ == '__main__':
    replicate()
