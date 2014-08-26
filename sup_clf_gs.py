#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: sup_clf_gs.py
# date: Thu August 14 10:06 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""sup_clf_gs: gridsearch to determine hyperparameters

"""

from __future__ import division
import os.path as path
import warnings
import cPickle as pickle


import numpy as np

import scipy

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.cross_validation import train_test_split
from scikits.audiolab import wavread
import spectral

import data


_wav_cache = {}
def _load_wav(fname, fs_=16000):
    key = fname
    if not key in _wav_cache:
        sig, fs, _ = wavread(fname)
        if fs_ != fs:
            raise ValueError('sampling rate should be {0}, not {1}. '
                             'please resample.'.format(fs_, fs))
        if len(sig.shape) > 1:
            warnings.warn('stereo audio: merging channels')
            sig = (sig[:, 0] + sig[:, 1]) / 2
        _wav_cache[key] = sig
    return _wav_cache[key]

def _load_spec(fname, nfilt, frate, highpass, fs=16000):
    sig = _load_wav(fname, fs)
    if highpass:
        sig = hpfilter(sig, fs, highpass)
    encoder = spectral.Spectral(nfilt=nfilt, fs=fs, wlen=0.025,
                                frate=frate, compression='log',
                                do_dct=False, do_deltas=False,
                                do_deltasdeltas=False)
    return encoder.transform(sig)

def hpfilter(sig, fs, cutoff, order=5):
    cutoff = cutoff / (0.5 * fs)
    b, a = scipy.signal.butter(order, cutoff, btype='highpass')
    return scipy.signal.lfilter(b, a, sig)


class AudioLoader(TransformerMixin, BaseEstimator):
    def __init__(self, highpass=None, nfilt=40, stacksize=30, normalize=True):
        self.nfilt = nfilt
        self.highpass = highpass
        self.stacksize = stacksize
        self.normalize = normalize

    def transform(self, X, y=None, **transform_params):
        frate = 100
        specs = {}
        for fname in X[:, 1]:
            specs[fname] = _load_spec(fname, self.nfilt, frate, self.highpass)

        nsamples = X.shape[0]
        X_ = np.empty((nsamples, self.stacksize * self.nfilt),
                      dtype=np.double)
        for idx, fragment in enumerate(X):
            _, filename, start, _ = fragment
            X_[idx] = data.stack_from_frame(specs[filename],
                                            int(float(start) * frate),
                                            self.stacksize)
        if self.normalize:
            X_ = (X_ - X_.mean(0)) / X_.std(0)
        return X_

    def fit(self, X, y=None, **fit_params):
        return self

def _load_files_labels(monkey, include_noise=False):
    if monkey == 'Blue_Fuller':
        dirname = 'Blue_monkeys_Fuller'
        annot = data.reduced_annotation(dirname,
                                        include_noise=include_noise)
        remap = {'A': 'p', 'PY': 'p', 'KA': 'h', 'KATR': 'h', 'NOISE': 'NOISE'}

        labelset_orig = sorted(list(set(f.mark for fname in annot
                                   for f in annot[fname])))
        labelset_remapped = sorted(list(set(remap.itervalues())))
        label2idx = {label: dict(zip(labelset_remapped,
                                 range(len(labelset_remapped))))[remap[label]]
                     for label in labelset_orig}
        X = np.array([(monkey,
                       path.join(data.BASEDIR, dirname, 'audio',
                                 f.filename + '.wav'),
                       '{0:.6f}'.format(f.interval.start),
                       '{0:.6f}'.format(f.interval.end))
                      for fname in annot for f in annot[fname]])
        y = np.fromiter((label2idx[f.mark]
                         for fname in annot for f in annot[fname]),
                        dtype=np.uint8)
        labels = labelset_remapped
    elif monkey == 'Blue_merged':
        dirname_murphy = 'Blue_monkeys'
        dirname_fuller = 'Blue_monkeys_Fuller'
        annot_murphy = data.reduced_annotation(dirname_murphy,
                                               include_noise=include_noise)
        annot_fuller = data.reduced_annotation(dirname_fuller,
                                               include_noise=include_noise)
        remap = {'A': 'p', 'PY': 'p', 'KA': 'h', 'KATR': 'h', 'NOISE': 'NOISE'}
        labelset_murphy = sorted(list(set(f.mark for fname in annot_murphy
                                          for f in annot_murphy[fname])))
        label2idx = {label: idx
                     for idx, label in enumerate(labelset_murphy)}
        X = np.vstack((np.array([('Blue_monkeys',
                                  path.join(data.BASEDIR, dirname_murphy,
                                            'audio', f.filename + '.wav'),
                                  '{0:.6f}'.format(f.interval.start),
                                  '{0:.6f}'.format(f.interval.end))
                                 for fname in annot_murphy
                                 for f in annot_murphy[fname]]),
                       np.array([('Blue_monkeys_Fuller',
                                  path.join(data.BASEDIR, dirname_fuller,
                                            'audio', f.filename + '.wav'),
                                  '{0:.6f}'.format(f.interval.start),
                                  '{0:.6f}'.format(f.interval.end))
                                 for fname in annot_fuller
                                 for f in annot_fuller[fname]])))
        y = np.hstack((np.fromiter((label2idx[f.mark]
                                    for fname in annot_murphy
                                    for f in annot_murphy[fname]),
                                   dtype=np.uint8),
                       np.fromiter((label2idx[remap[f.mark]]
                                    for fname in annot_fuller
                                    for f in annot_fuller[fname]),
                                   dtype=np.uint8)))

        labels = labelset_murphy
    elif monkey == 'all':
        X = None
        y = None
        labels = ['NOISE'] if include_noise else []
        label2idx = None
        for m in ['colobus', 'Titi_monkeys', 'Blue_merged']:
            X_, y_, labels_ = _load_files_labels(m, include_noise=include_noise)

            if not X is None:
                X = np.vstack((X, X_))

                label2idx_ = {label: idx for idx, label in enumerate(labels_)}
                labelinds = {label: np.nonzero(y_==label2idx_[label])
                             for label in labels_}
                labels_clean_ = [label for label in labels_
                                 if label != 'NOISE']
                remap_idx = dict(zip(labels_clean_,
                                     range(len(labels),
                                           len(labels) + len(labels_clean_))))
                if include_noise:
                    remap_idx['NOISE'] = 0
                y_new = np.zeros(y_.shape, dtype=np.uint8)
                for label, inds in labelinds.iteritems():
                    y_new[inds] = remap_idx[label]
                y = np.hstack((y, y_new))

                labels.extend([m + '-' + label for label in labels_clean_])
            else:
                X = X_
                y = y_
                labels.extend([m + '-' + label for label in labels_
                               if label != 'NOISE'])
            label2idx = {label: idx for idx, label in enumerate(labels)}
    else:
        if monkey == 'Blue_Murphy':
            monkey = 'Blue_monkeys'
        annot = data.reduced_annotation(monkey, include_noise=include_noise)
        labels = sorted(list(set(f.mark for fname in annot
                                   for f in annot[fname])))
        label2idx = {l:i for i, l in enumerate(labels)}
        X = np.array([(monkey,
                       path.join(data.BASEDIR, monkey, 'audio',
                                 f.filename + '.wav'),
                       '{0:.6f}'.format(f.interval.start),
                       '{0:.6f}'.format(f.interval.end))
                      for fname in annot for f in annot[fname]])
        y = np.fromiter((label2idx[f.mark]
                         for fname in annot for f in annot[fname]),
                        dtype=np.uint8)
    return X, y, labels


if __name__ == '__main__':
    for monkey in ['Titi_monkeys', 'colobus', 'Blue_Fuller', 'Blue_Murphy',
                   'Blue_merged', 'all']:
        print
        print monkey
        print
        X, y, labels = _load_files_labels(monkey, include_noise=False)

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipeline = Pipeline([('data', AudioLoader(highpass=None,
                                                  normalize=True)),
                             ('selector', SelectPercentile(f_classif)),
                             ('svm', SVC(kernel='rbf', gamma=1e-5))])

        paramdist = {'data__nfilt': scipy.stats.randint(10, 50),
                     'data__stacksize': scipy.stats.randint(11, 51),
                     'selector__percentile': scipy.stats.randint(10, 100),
                     'svm__C': np.logspace(0, 2, 50)}

        clf = RandomizedSearchCV(pipeline, paramdist, verbose=3, n_iter=1000,
                                 cv=StratifiedKFold(y_train, n_folds=2),
                                 n_jobs=35)
        clf.fit(X_train, y_train)
        with open(path.join(data.BASEDIR, 'sup_clf_rand_{0}.pkl'.format(monkey)),
                  'wb') as fid:
            pickle.dump(clf.best_params_, fid, -1)

        y_pred = clf.predict(X_test)
        with open(path.join(data.BASEDIR, 'sup_clf_rand_results_{0}.pkl'.format(monkey)),
                  'wb') as fid:
            pickle.dump((y_test, y_pred, labels), fid, -1)
        # _wav_cache = {}
