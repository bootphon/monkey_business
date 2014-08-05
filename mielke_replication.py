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
from pprint import pformat
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.special import expit

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics

from data import BASEDIR, load_data_stacked


MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus', 'Blue_monkeys_Fuller']


def load_all_monkeys():
    """Load stacked data for all monkeys.

    :return
      X: dict from monkeyname to ndarray containing the stacked audio
      y: dict from monkeyname to ndarray containing the labels (as ints)
      labelset: dict from monkeyname to sorted list of call names
    """
    _memo_fname = path.join(BASEDIR, 'monkey_calls_stacked.pkl')
    if path.exists(_memo_fname):
        print 'loading data from:', _memo_fname
        with open(_memo_fname, 'rb') as fid:
            X, y, labelset = pickle.load(fid)
    else:
        print 'loading data from individual files...'
        labelset = {}
        X = {}
        y = {}
        for monkey in MONKEYS:
            print '  ', monkey
            X_, y_, labelset_ = load_data_stacked(monkey)
            labelset[monkey] = labelset_
            X[monkey] = X_
            y[monkey] = y_

        with open(_memo_fname, 'wb') as fid:
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

def print_cm(stream, cm, target_names=None, vert_labels=False):
    """pretty print the confusion matrix to stream"""
    def print_vert(stream, strings, spacing=1, offset=0):
        lsize = max(len(x) for x in strings)
        strings = ['{0:s}'.format(l.rjust(lsize)) for l in strings]
        for line in range(lsize):
            print >>stream, ' ' * offset + (' ' * spacing).join(s[line] for s in strings)
    nlabels = cm.shape[0]
    esize = int(np.maximum(np.max(np.floor(np.log10(cm))+1), 0))
    if target_names is None:
        target_names = map(str, range(nlabels))
    lsize = max(len(x) for x in target_names)
    names = ['{0:s}'.format(n.rjust(lsize)) for n in target_names]
    if vert_labels:
        print_vert(stream, names, spacing=esize, offset=lsize+esize)
        print >>stream, ''
    for y_idx in range(cm.shape[0]):
        name = '{0:s}'.format(names[y_idx].rjust(lsize))
        vline = ' '.join(str(cm[y_idx, x_idx]).rjust(esize)
                         for x_idx in range(cm.shape[1]))
        print >>stream, name + ' ' + vline


def classification_by_monkey(X, y, labelset, stream, n_folds=5,
                             do_gridsearch=True,
                             verbose=True):
    for monkey in MONKEYS:
        if verbose:
            print '-' * len(monkey)
            print monkey
            print '-' * len(monkey)

        print >>stream, '***', monkey

        y_true = None
        y_pred = None
        pvals = None
        print >>stream, '\n**** Cross-validation scores\n'

        for fold in range(n_folds):
            if verbose:
                print '  FOLD:', fold
            X_train, X_test, y_train, y_test = train_test_split(X[monkey],
                                                                y[monkey],
                                                                test_size=0.1)
            if verbose:
                print 'training classifier...'
            if do_gridsearch:
                clf = GridSearchCV(SVC(),
                                   [{'kernel': ['rbf'],
                                     'gamma': [1e-4],
                                     'C': np.logspace(0, 2, 10)},
                                   ],
                                   cv=3, score_func=metrics.f1_score,
                                   verbose=1 if verbose else 0, n_jobs=-1)
            else:
                clf = SVC(kernel='rbf', C=1)
            clf.fit(X_train, y_train)

            if do_gridsearch:
                print >>stream, 'FOLD:', fold, clf.best_score_
                print >>stream, pformat(clf.best_params_)

            if verbose:
                print 'predicting class labels...'
            if y_true is None:
                y_true = y_test
                y_pred = clf.predict(X_test)
                pvals = expit(clf.decision_function(X_test))
            else:
                y_true = np.hstack((y_true, y_test))
                y_pred = np.hstack((y_pred, clf.predict(X_test)))
                pvals = np.hstack((pvals, expit(clf.decision_function(X_test))))
        print >>stream, '\n**** Classification report\n'
        print >>stream, metrics.classification_report(y_true, y_pred,
                                            target_names=labelset[monkey])
        print >>stream, '\n**** Confusion matrix\n'
        print_cm(stream, metrics.confusion_matrix(y_true, y_pred),
                 labelset[monkey])
        print >>stream, ''
        stream.flush()
        with open('clf_by_monkey_{0}.pkl'.format(monkey), 'wb') as fid:
            pickle.dump((y_true, y_pred, pvals, labelset[monkey]), fid, -1)

def classification_across_monkey(X, y, labelset, stream, n_folds=5,
                                 do_gridsearch=True,
                                 verbose=True):
    X_comb, y_comb, labelset_comb = combine_labels(X, y, labelset)

    print >>stream, '*** Cross-validation scores\n'
    y_true = None
    y_pred = None
    pvals = None

    for fold in range(n_folds):
        X_train, X_test, y_train, y_test = train_test_split(X_comb, y_comb,
                                                            test_size=0.1)
        if verbose:
            print 'training classifier...'
        if do_gridsearch:
            clf = GridSearchCV(SVC(),
                               [{'kernel':['rbf'],
                                 'gamma': [1e-5],
                                 'C': np.logspace(1, 3, 10)},
                                # {'kernel': ['linear'],
                                #  'C': np.logspace(0, 3, 10)}],
                                ],
                               cv=3, score_func=metrics.f1_score,
                               verbose=1 if verbose else 0, n_jobs=-1)
        else:
            clf = SVC(kernel='rbf', C=1)
        clf.fit(X_train, y_train)
        if do_gridsearch:
            print >>stream, 'FOLD:', fold, clf.best_score_
            print >>stream, pformat(clf.best_params_)

        if y_true is None:
            y_true = y_test
            y_pred = clf.predict(X_test)
            pvals = expit(clf.decision_function(X_test))
        else:
            y_true = np.hstack((y_true, y_test))
            y_pred = np.hstack((y_pred, clf.predict(X_test)))
            pvals = np.hstack((pvals, expit(clf.decision_function(X_test))))

    print >>stream, '\n*** Classification report\n'
    print >>stream, metrics.classification_report(y_true, y_pred,
                                        target_names=labelset_comb)
    print >>stream, '\n*** Confusion matrix\n'
    print_cm(stream, metrics.confusion_matrix(y_true, y_pred),
             labelset_comb)
    print >>stream, ''
    stream.flush()
    with open('clf_across_monkey.pkl', 'wb') as fid:
        pickle.dump((y_true, y_pred, pvals, labelset_comb), fid, -1)


def classification_by_species(X, y, labelset, stream, n_folds=5,
                              do_gridsearch=True, verbose=True):
    X_comb, y_comb, labelset_comb = make_monkey_set(X, y, labelset)

    y_true = None
    y_pred = None
    pvals = None
    print >>stream, '*** Cross-validation scores\n'
    for fold in range(n_folds):
        if verbose:
            print '  FOLD:', fold
        X_train, X_test, y_train, y_test = train_test_split(X_comb, y_comb,
                                                            test_size=0.1)
        if do_gridsearch:
            clf = GridSearchCV(SVC(),
                               [{'kernel': ['rbf'],
                                 'gamma': np.logspace(-5, -3, 3),
                                 'C': np.logspace(0, 3, 10)}],
                               cv=3, score_func=metrics.f1_score,
                               verbose=1 if verbose else 0, n_jobs=-1)
        else:
            clf = SVC(kernel='rbf', C=1)
        clf.fit(X_train, y_train)

        if do_gridsearch:
            print >>stream, 'FOLD:', fold, clf.best_score_
            print >>stream, pformat(clf.best_params_)
        if y_true is None:
            y_true = y_test
            y_pred = clf.predict(X_test)
            pvals = expit(clf.decision_function(X_test))
        else:
            y_true = np.hstack((y_true, y_test))
            y_pred = np.hstack((y_pred, clf.predict(X_test)))
            pvals = np.hstack((pvals, expit(clf.decision_function(X_test))))

    print >>stream, '\n*** Classification report\n'
    print >>stream, metrics.classification_report(y_true, y_pred,
                                                     target_names=labelset_comb)
    print >>stream, '\n*** Confusion matrix\n'
    print_cm(stream, metrics.confusion_matrix(y_true, y_pred),
             labelset_comb)
    print >>stream, ''
    stream.flush()
    with open('clf_species.pkl', 'wb') as fid:
        pickle.dump((y_true, y_pred, pvals, labelset_comb), fid, -1)


def replicate(resultfile, n_folds=5, do_gridsearch=True, verbose=True):
    X, y, labelset = load_all_monkeys()

    with open(resultfile, 'w') as stream:
        print >>stream, '* Replication of Mielke & Zuberbuehler'
        print >>stream, """
Replication of some of the results described in Mielke, Zuberbuehler (2013),
"A method for automated individual, species and call type recognition in
free-ranging animals", Animal Behaviour 86, pp 475--482

The document lists the results of multiclass classification only on
pre-determined intervals of audio. Manually labeled intervals were extracted
from the audio recordings of monkey calls. Labels with less than 50 instances
were discarded. Classification was performed with a Support Vector Classifier
with a radial basis function kernel.

A grid search with 3-fold crossvalidation on the training set was performed to
tune the hyperparameters $C$ and $\gamma$. Scores are reported on the average
of 5 independent splits of the data into training and test sets.


This file was automatically generated by running `mielke_replication.py`.

"""
        stream.flush()
        # 1. classify calls per monkey
        if verbose:
            print '---------------------------'
            print '1. CLASSIFICATION BY MONKEY'
            print '---------------------------'

        print >>stream, '** CLASSIFICATION BY MONKEY'

        classification_by_monkey(X, y, labelset, stream, n_folds=n_folds,
                                 do_gridsearch=do_gridsearch,
                                 verbose=verbose)

        # 2. classify over all monkeys and calls
        if verbose:
            print '-------------------------------'
            print '2. CLASSIFICATION ACROSS MONKEY'
            print '-------------------------------'
        print >>stream, '** CLASSIFICATION ACROSS MONKEYS'
        classification_across_monkey(X, y, labelset, stream, n_folds=n_folds,
                                    do_gridsearch=do_gridsearch,
                                    verbose=verbose)

        # 3. classify the monkeys
        if verbose:
            print '----------------------------'
            print '3. CLASSIFICATION BY SPECIES'
            print '----------------------------'
        print >>stream, '** CLASSIFICATION BY SPECIES'
        classification_by_species(X, y, labelset, stream, n_folds=n_folds,
                                  do_gridsearch=do_gridsearch,
                                  verbose=verbose)


if __name__ == '__main__':
    replicate('mielke_results.org')
