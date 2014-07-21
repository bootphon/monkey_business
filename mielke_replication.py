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

import os
import os.path as path
import fnmatch
from collections import namedtuple, defaultdict
from itertools import imap
import warnings
import cPickle as pickle

from scikits.audiolab import wavread
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import sklearn.metrics as metrics

import numpy as np

import spectral
from textgrid import TextGrid

MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus']
# MONKEYS = ['Titi_monkeys']

BASEDIR = path.join(os.environ['HOME'], 'data', 'monkey_sounds')

Interval = namedtuple('Interval', ['start', 'end'])
Fragment = namedtuple('Fragment', ['filename', 'interval', 'mark'])


def rglob(rootdir, pattern):
    for root, _, files in os.walk(rootdir):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield path.join(root, basename)


def get_annotation(monkey):
    monkeydir = path.join(BASEDIR, monkey)
    annot = defaultdict(list)
    for tgfile in rglob(path.join(monkeydir, 'textgrids'), '*.TextGrid'):
        filename = path.splitext(path.basename(tgfile))[0]
        if not path.exists(path.join(monkeydir, 'audio', filename + '.wav')):
            continue
        tg = TextGrid.read(tgfile)
        tier = tg.tiers[0]
        for interval in tier:
            if interval.mark.strip() != '':
                fragment = Fragment(filename,
                                    Interval(interval.start - tier.start,
                                             interval.end - tier.start),
                                    interval.mark.strip())
                annot[filename].append(fragment)
    return annot

def load_data(monkey, verbose=True):
    annot = get_annotation(monkey)

    # do some filtering to get rid of weird labels in titi monkeys
    counts = defaultdict(int)
    for fragments in annot.itervalues():
        for fragment in fragments:
            counts[fragment.mark] += 1


    # labelset = sorted(m for m in counts if counts[m] > 50)

    annot = {k: [f for f in v if counts[f.mark] > 50] for k, v in annot.iteritems()}
    labelset = sorted(list(set(f.mark for v in annot.itervalues() for f in v)))

    # labelset = sorted(counts.keys())
    label2idx = dict(zip(labelset, range(len(labelset))))
    nsamples = sum(imap(len, annot.itervalues()))

    if verbose:
        print monkey
        print '  number of samples:', nsamples
        print '  unique labels:'
        for label in labelset:
            print '    {0}: {1}'.format(label, counts[label])

    nfilt = 40
    wlen = 0.025
    frate = 100
    fs = 16000
    encoder = spectral.Spectral(nfilt=nfilt, fs=fs, wlen=wlen, frate=frate,
                                compression='log', nfft=1024,
                                do_dct=False, do_deltas=False, do_deltasdeltas=False)
    NFRAMES = 30
    X = np.empty((nsamples, NFRAMES*nfilt), dtype=np.double)
    y = np.empty((nsamples,), dtype=np.uint8)
    idx = 0
    for fname in annot:
        wav = path.join(BASEDIR, monkey, 'audio', fname + '.wav')
        if not path.exists(wav):
            raise ValueError('missing wave file: {0}'.format(fname))
        sig, fs_, _ = wavread(wav)
        if fs_ != fs:
            raise ValueError('sampling rate should be 16kHz, not {0}'.format(fs_))
        if len(sig.shape) > 1:
            # should not happen, so warn
            warnings.warn('stereo audio found, please merge channels')
            sig = (sig[:, 0] + sig[:, 1]) / 2
        spec = encoder.transform(sig)

        for fragment in annot[fname]:
            # if not fragment.mark in labelset:
            #     continue
            start_frame = int(fragment.interval.start * frate)
            if start_frame + NFRAMES >= spec.shape[0]:
                s = spec[start_frame: start_frame + NFRAMES]
                extra_zeros = np.zeros((start_frame + NFRAMES - spec.shape[0],
                                        nfilt),
                                       dtype=s.dtype)
                x = np.vstack((s, extra_zeros))
                X[idx] = np.hstack(x)
            else:
                X[idx] = np.hstack(spec[start_frame: start_frame + NFRAMES])
            try:
                y[idx] = label2idx[fragment.mark]
            except KeyError as e:
                print fragment
                print
                print label2idx
                print
                print labelset
                print
                raise e

            idx += 1
    return X, y, labelset

def load_all():
    _memo_fname = 'monkey_data.pkl'
    if path.exists(_memo_fname):
        with open(_memo_fname, 'rb') as fid:
            X, y, labelset = pickle.load(fid)
    else:
        labelset = {}
        X = {}
        y = {}
        for monkey in MONKEYS:
            X_, y_, labelset_ = load_data(monkey)
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
    idx2comblabel = dict(zip(range(len(combined_labels)), combined_labels))

    X_comb = None
    y_comb = None
    for comb_label in combined_labels:
        monkey, label = comb_label.split('-')
        label2idx = dict(zip(labelset[monkey], range(len(labelset[monkey]))))
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
    X, y, labelset = load_all()

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
