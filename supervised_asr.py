#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: supervised_asr.py
# date: Wed July 23 19:19 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""supervised_asr:

"""

from __future__ import division

import os.path as path
import cPickle as pickle
import time
import sys
import data
import spectral

import numpy as np
import scipy.stats
from sklearn.preprocessing import StandardScaler

MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus']
# MONKEYS = ['Blue_monkeys']


def load_all_intervals(nfilt=40, stacksize=30, highpass=200):
    print 'loading annotations...',
    annot_all = {}
    t0 = time.time()
    for monkey in MONKEYS:
        annot_all[monkey] = data.reduced_annotation(monkey)
    print 'done. time taken: {0:.3f}s'.format(time.time() - t0)
    sys.stdout.flush()

    print 'splitting files...',
    sys.stdout.flush()
    t0 = time.time()
    train_test_files = {k: data.train_test_split_files(annot_all[k])
                        for k in annot_all}
    print 'done. time taken: {0:.3f}s'.format(time.time() - t0)
    sys.stdout.flush()

    frate = 100
    encoder = spectral.Spectral(nfilt=nfilt, fs=16000, wlen=0.025, frate=frate,
                                compression='log', nfft=1024, do_dct=False,
                                do_deltas=False, do_deltasdeltas=False)

    X_train = {}
    X_test = {}
    y_train = {}
    y_test = {}
    labelset = {}
    for monkey in MONKEYS:
        print monkey
        sys.stdout.flush()

        train_files, test_files = train_test_files[monkey]
        annot = annot_all[monkey]

        annot_train = {fname: annot[fname] for fname in train_files}
        print '  loading train data...',
        sys.stdout.flush()
        t0 = time.time()
        X_train_, y_train_, labels_train = data.load_data_stacked_annot(monkey,
                                                          annot_train,
                                                          encoder, stacksize, highpass=highpass)
        print 'done. time taken: {0:.3f}s'.format(time.time() - t0)
        sys.stdout.flush()


        print '  loading test data...',
        sys.stdout.flush()
        t0 = time.time()
        annot_test = {fname: annot[fname] for fname in test_files}
        X_test_, y_test_, labels_test = data.load_data_stacked_annot(monkey,
                                                    annot_test,
                                                         encoder, stacksize, highpass=highpass)
        print 'done. time taken: {0:.3f}s'.format(time.time() - t0)
        sys.stdout.flush()

        # print '  scaling...',
        # sys.stdout.flush()
        # t0 = time.time()
        # scaler = StandardScaler().fit(np.vstack((X_train_, X_test_)))

        # X_train[monkey] = scaler.transform(X_train_)
        # y_train[monkey] = y_train_
        # X_test[monkey] = scaler.transform(X_test_)
        # y_test[monkey] = y_test_
        # print 'done. time taken: {0:.3f}s'.format(time.time() - t0)

        X_train[monkey] = X_train_
        y_train[monkey] = y_train_
        X_test[monkey] = X_test_
        y_test[monkey] = y_test_

        assert(labels_test == labels_train)
        labelset[monkey] = labels_train

    return X_train, X_test, y_train, y_test, labelset
