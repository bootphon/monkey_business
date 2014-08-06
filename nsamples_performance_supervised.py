#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: manlabeled_nsamples.py
# date: Tue August 05 03:05 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""nsamples_performance_supervised: investigate the number of manually labeled examples
needed for performance. replication of mielke_replication but with variable
number of training samples

"""

from __future__ import division
import os
import os.path as path
import cPickle as pickle
from pprint import pformat
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.special import expit

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics

from data import BASEDIR, load_data_stacked

MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus', 'Blue_monkeys_Fuller']

import mielke_replication as mr


def stratified_sample(X, y, n):
    """Sample n elements from X, y, preserving class balance

    Arguments:
    :param X:
    :param y:
    :param n:
    :param seed:
    """
    res = X.shape[0] - n
    classes, y_indices = np.unique(y, return_inverse=True)
    cls_count = np.bincount(y_indices)
    p_i = cls_count / len(y)
    n_i = np.maximum(np.round(n * p_i).astype(int),
                     4)  # minimum of four samples per class
    t_i = np.minimum(cls_count - n_i, np.round(res * p_i).astype(int))
    inds = []
    for i, cls in enumerate(classes):
        permutation = np.random.permutation(n_i[i]+t_i[i])
        cls_i = np.where((y == cls))[0][permutation]
        inds.extend(cls_i[:n_i[i]])
    inds = np.random.permutation(inds)
    return X[inds], y[inds]


def classification_by_monkey(X, y, labelset, param_grid,
                             n_steps=20, n_folds_test=20, n_folds_gridsearch=5,
                             verbose=True):
    for monkey in MONKEYS:
        if verbose:
            print monkey
        scores = np.zeros((n_steps, 6))
        min_nsamples = len(labelset[monkey]) * 2
        for step in range(n_steps):
            y_true = None
            y_pred = None
            avg_nsamples = 0
            for fold in range(n_folds_test):
                if verbose:
                    print '\r  step: {0:3d}/{1:3d}, fold: {2:3d}'\
                        .format(step+1, n_steps, fold+1),
                X_train, X_test, y_train, y_test = \
                    train_test_split(X[monkey], y[monkey], test_size=0.1)

                steps = np.linspace(min_nsamples,
                                    X_train.shape[0],
                                    n_steps).astype(int)
                n_train = steps[step]

                X_train, y_train = stratified_sample(X_train, y_train, n_train)
                avg_nsamples += X_train.shape[0]
                clf = GridSearchCV(SVC(),
                                   param_grid,
                                   cv=StratifiedKFold(y_train,
                                                      n_folds=n_folds_gridsearch),
                                   score_func=metrics.accuracy_score,
                                   verbose=0 if verbose else 0, n_jobs=-1)
                clf.fit(X_train, y_train)
                if y_true is None:
                    y_true = y_test
                    y_pred = clf.predict(X_test)
                else:
                    y_true = np.hstack((y_true, y_test))
                    y_pred = np.hstack((y_pred, clf.predict(X_test)))
            avg_nsamples /= n_folds
            scores[step] = np.hstack(([avg_nsamples, y_true.shape[0]],
                metrics.precision_recall_fscore_support(y_true, y_pred,
                                                        average='weighted')[:-1],
                                      [metrics.accuracy_score(y_true, y_pred)]))
        np.savetxt('results/clf_by_nsamples_{0}.txt'.format(monkey),
                   scores, fmt=['%.0f','%.0f', '%.3f','%.3f','%.3f', '%.3f'],
                   delimiter='\t',
                   header='nsamples\tsupport\tprecision\trecall\tfscore\taccuracy')
        if verbose:
            print

def classification_across_monkey(X, y, labelset, param_grid,
                                 n_steps=20, n_folds_test=20,
                                 n_folds_gridsearch=5,
                                 verbose=True):
    X, y, labelset = mr.combine_labels(X, y, labelset)

    if verbose:
        print 'classification across monkey'
    min_nsamples = len(labelset) * 2
    scores = np.zeros((n_steps, 6))
    for step in range(n_steps):
        y_true = None
        y_pred = None
        avg_nsamples = 0
        for fold in range(n_folds_test):
            if verbose:
                print '\r  step: {0:3d}/{1:3d}, fold: {2:3d}'\
                    .format(step+1, n_steps, fold+1),
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
            steps = np.linspace(min_nsamples,
                                X_train.shape[0],
                                n_steps).astype(int)
            n_train = steps[step]
            X_train, y_train = stratified_sample(X_train, y_train, n_train)
            avg_nsamples += X_train.shape[0]
            clf = GridSearchCV(SVC(),
                               param_grid,

                               cv=StratifiedKFold(y_train,
                                                  n_folds=n_folds_gridsearch),
                               score_func=metrics.accuracy_score,
                               verbose=0, n_jobs=-1)
            clf.fit(X_train, y_train)
            if y_true is None:
                y_true = y_test
                y_pred = clf.predict(X_test)
            else:
                y_true = np.hstack((y_true, y_test))
                y_pred = np.hstack((y_pred, clf.predict(X_test)))
        avg_nsamples /= n_folds
        scores[step] = np.hstack(([avg_nsamples, y_true.shape[0]],
                                  metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                          average='weighted')[:-1],
                                  [metrics.accuracy_score(y_true, y_pred)]))
    np.savetxt('results/clf_across_nsamples.txt',
               scores, fmt=['%.0f','%.0f', '%.3f','%.3f','%.3f', '%.3f'],
               delimiter='\t',
               header='nsamples\tsupport\tprecision\trecall\tfscore\taccuracy')
    if verbose:
        print

if __name__ == '__main__':
    X, y, labelset = mr.load_all_monkeys()
    from svc_param_grid import param_grid

    n_steps = 25
    n_folds = 10
    try:
        os.makedirs('results')
    except OSError:
        pass
    classification_by_monkey(X, y, labelset, param_grid,
                             n_steps=n_steps,
                             n_folds_test=n_folds)

    classification_across_monkey(X, y, labelset, param_grid,
                                 n_steps=n_steps,
                                 n_folds_test=n_folds)
