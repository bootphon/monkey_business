#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: transcriber.py
# date: Mon August 11 18:38 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""transcriber: train and run transcription over entire audio files


"""

from __future__ import division
import os.path as path
import cPickle as pickle
import warnings

import numpy as np
import scipy.signal
import scipy.special
import scipy.stats

from scikits.audiolab import wavread
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import spectral
import data
from resampling import Resampler


THETAS       = {'Blue_monkeys': 0.5066,
                'Blue_monkeys_Fuller': 0.5062,
                'Titi_monkeys': 0.5007,
                'colobus': 0.5060}

_wav_cache = {}
def _load_wav(monkey, fname, fs_=16000):
    key = (monkey, fname)
    if not key in _wav_cache:
        sig, fs, _ = wavread(path.join(data.BASEDIR, monkey, 'audio',
                                       fname + '.wav'))
        if fs_ != fs:
            raise ValueError('sampling rate should be {0}, not {1}. '
                             'please resample.'.format(fs_, fs))
        if len(sig.shape) > 1:
            warnings.warn('stereo audio: merging channels')
            sig = (sig[:, 0] + sig[:, 1]) / 2
        _wav_cache[key] = sig
    return _wav_cache[key]

_spec_cache = {}
def _load_spec(monkey, fname, nfilt, frate, highpass, fs=16000):
    key = (monkey, fname, nfilt, frate, highpass)
    if not key in _spec_cache:
        sig = _load_wav(monkey, fname)
        if highpass:
            sig = hpfilter(sig, fs, highpass)
        encoder = spectral.Spectral(nfilt=nfilt, fs=fs, wlen=0.025,
                                    frate=frate, compression='log',
                                    do_dct=False, do_deltas=False,
                                    do_deltasdeltas=False)
        _spec_cache[key] = encoder.transform(sig)
    return _spec_cache[key]

_vad_cache = {}
def _load_vad(monkey):
    theta = THETAS[monkey]
    key = monkey
    if not key in _vad_cache:
        with open(path.join(data.BASEDIR, 'mad_{0}.pkl'.format(monkey))) as fid:
            pred_vad = pickle.load(fid)
        for fname in pred_vad:
            pred_vad[fname] = scipy.special.expit(pred_vad[fname]) >= theta
        _vad_cache[key] = pred_vad
    return _vad_cache[key]


def _load_files_labels(monkey, under_sample=True, under_sampling_ratio=1.0,
                       shuffle=True, frate=100):
    if monkey == 'Blue_Fuller':
        dirname = 'Blue_monkeys_Fuller'
        annot = data.reduced_annotation(dirname, include_noise=True)
        remap = {'A': 'p', 'PY': 'p', 'KA': 'h', 'KATR': 'h', 'NOISE': 'NOISE'}

        labelset_orig = sorted(list(set(f.mark for fname in annot
                                   for f in annot[fname])))
        labelset_remapped = sorted(list(set(remap.itervalues())))
        label2idx = {label: dict(zip(labelset_remapped,
                                 range(len(labelset_remapped))))[remap[label]]
                     for label in labelset_orig}
        X = []
        y = []
        for fname in annot:
            for idx, fragment in enumerate(annot[fname]):
                start = fragment.interval.start
                end = fragment.interval.end
                start_fr = int(start * frate)
                end_fr = int(end * frate)
                if idx == len(annot[fname]) - 1:
                    end_fr -= 10
                    if end_fr <= start_fr:
                        continue
                for n in range(start_fr, end_fr):
                    X.append((dirname, fname, str(n),
                              '{0:.3f}'.format(start+n/frate),
                              '{0:.3f}'.format(start+(n+1)/frate)))
                    y.append(label2idx[fragment.mark])
        X = np.array(X)
        y = np.array(y)
        labels = labelset_remapped
    elif monkey == 'Blue_merged':
        dirname_murphy = 'Blue_monkeys'
        dirname_fuller = 'Blue_monkeys_Fuller'
        annot_murphy = data.reduced_annotation(dirname_murphy,
                                               include_noise=True)
        annot_fuller = data.reduced_annotation(dirname_fuller,
                                               include_noise=True)
        remap = {'A': 'p', 'PY': 'p', 'KA': 'h', 'KATR': 'h', 'NOISE': 'NOISE'}
        labelset_murphy = sorted(list(set(f.mark for fname in annot_murphy
                                          for f in annot_murphy[fname])))
        label2idx = {label: idx
                     for idx, label in enumerate(labelset_murphy)}
        X = []
        y = []

        for fname in annot_murphy:
            for idx, fragment in enumerate(annot_murphy[fname]):
                start = fragment.interval.start
                end = fragment.interval.end
                start_fr = int(start * frate)
                end_fr = int(end * frate)
                if idx == len(annot_murphy[fname]) - 1:
                    end_fr -= 10
                    if end_fr <= start_fr:
                        continue
                for n in range(start_fr, end_fr):
                    X.append((dirname_murphy, fname, str(n),
                              '{0:.3f}'.format(start+n/frate),
                              '{0:.3f}'.format(start+(n+1)/frate)))
                    y.append(label2idx[fragment.mark])
        for fname in annot_fuller:
            for idx, fragment in enumerate(annot_fuller[fname]):
                start = fragment.interval.start
                end = fragment.interval.end
                start_fr = int(start * frate)
                end_fr = int(end * frate)
                if idx == len(annot_fuller[fname]) - 1:
                    end_fr -= 10
                    if end_fr <= start_fr:
                        continue
                for n in range(start_fr, end_fr):
                    X.append((dirname_fuller, fname, str(n),
                              '{0:.3f}'.format(start+n/frate),
                              '{0:.3f}'.format(start+(n+1)/frate)))
                    y.append(label2idx[remap[fragment.mark]])
        X = np.array(X)
        y = np.array(y)
        labels = labelset_murphy
    elif monkey == 'all':
        X = None
        y = None
        labels = ['NOISE']
        label2idx = None
        for m in ['colobus', 'Titi_monkeys', 'Blue_merged']:
            X_, y_, labels_ = _load_files_labels(m, shuffle=False,
                                                 under_sample=False)
            if not X is None:
                X = np.vstack((X, X_))
                label2idx_ = {label: idx for idx, label in enumerate(labels_)}
                labelinds = {label: np.nonzero(y_==label2idx_[label])
                             for label in labels_}
                labels_clean_ = [label for label in labels_
                                 if label != 'NOISE']
                remap_idx = dict(zip(labels_clean_,
                                     range(len(labels),
                                           len(labels + len(labels_clean_)))))
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
        annot = data.reduced_annotation(monkey, include_noise=True)
        labels = sorted(list(set(f.mark for fname in annot
                                 for f in annot[fname])))
        label2idx = {label: idx for idx, label in enumerate(labels)}
        X = []
        y = []
        for fname in annot:
            for idx, fragment in enumerate(annot[fname]):
                start = fragment.interval.start
                end = fragment.interval.end
                start_fr = int(start * frate)
                end_fr = int(end * frate)
                if idx == len(annot[fname]) - 1:
                    end_fr -= 10
                    if end_fr <= start_fr:
                        continue
                for n in range(start_fr, end_fr):
                    X.append((monkey, fname, str(n),
                              '{0:.3f}'.format(start+n/frate),
                              '{0:.3f}'.format(start+(n+1)/frate)))
                    y.append(label2idx[fragment.mark])
        X = np.array(X)
        y = np.array(y)
    if under_sample:
        X, y = Resampler(ratio=under_sampling_ratio).resample(X, y)
    if shuffle:
        inds = np.random.permutation(X.shape[0])
        X = X[inds]
        y = y[inds]
    return X, y, labels

class AudioLoader(TransformerMixin, BaseEstimator):
    def __init__(self, nfilt=40, highpass=None, normalize=True, stacksize=31):
        self.nfilt = nfilt
        self.highpass = highpass
        self.normalize = normalize
        self.stacksize = stacksize

    def transform(self, X, y=None, **transform_params):
        specs = {}
        for fname in np.unique(X[:, 1]):
            arr = _load_spec(monkey, fname, self.nfilt, FRATE, self.highpass)
            arr = np.vstack((np.zeros((self.stacksize//2, arr.shape[1])),
                             arr,
                             np.zeros((self.stacksize//2, arr.shape[1]))))
            specs[fname] = np.hstack(np.roll(arr, -i, 0)
                                     for i in xrange(self.stacksize)) \
                                         [:arr.shape[0] - self.stacksize + 1]

        nsamples = X.shape[0]
        X_ = np.empty((nsamples, self.stacksize * self.nfilt), dtype=np.double)
        for idx, (_, fname, frameno, _, _) in enumerate(X):
            X_[idx] = specs[fname][int(frameno)]
        if self.normalize:
            X_ = (X_ - X_.mean(0)) / X_.std(0)
        return X_

    def fit(self, X, y=None, **fit_params):
        return self

class VADLoader(TransformerMixin, BaseEstimator):
    def __init__(self, normalize=True, stacksize=31):
        self.normalize = normalize
        self.stacksize = stacksize
    def transform(self, X, y=None, **transform_params):
        vads = {}
        for monkey in np.unique(X[:, 0]):
            vads[monkey] = _load_vad(monkey)
        nsamples = X.shape[0]
        X_ = np.empty((nsamples, self.stacksize), dtype=np.double)
        for idx, (monkey, fname, frameno, _, _) in enumerate(X):
            X_[idx] = vads[monkey][fname][int(frameno)]
        if self.normalize:
            X_ = (X_ - X_.mean(0)) / X_.std(0)
        return X_

    def fit(self, X, y=None, **fit_params):
        return self

def hpfilter(sig, fs, cutoff, order=5):
    cutoff = cutoff / (0.5 * fs)
    b, a = scipy.signal.butter(order, cutoff, btype='highpass')
    return scipy.signal.lfilter(b, a, sig)


if __name__ == '__main__':
    UNDERSAMPLING_RATIO = 1.1
    FRATE = 100
    for monkey in ['Titi_monkeys', 'colobus', 'Blue_Fuller', 'Blue_Murphy',
                   'Blue_merged', 'all']:
        X, y, labels = _load_files_labels(monkey,
                                          under_sampling_ratio=UNDERSAMPLING_RATIO)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipeline = Pipeline([('data', FeatureUnion([('audio', AudioLoader()),
                                                    ('vad', VADLoader())])),
                             ('svm', SVC(kernel='rbf', gamma=1e-5, C=20))
                             ])
        paramdist = {'svm__C': np.logspace(0, 2, 50),
                     'data__vad__stacksize': scipy.stats.randint(11, 51),
                     'data__audio__stacksize': scipy.stats.randint(11, 51)}
        clf = RandomizedSearchCV(pipeline, paramdist, n_iter=500, verbose=1,
                                 cv=1,
                                 n_jobs=35)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        with open(path.join(data.BASEDIR,
                            'transcriber_rand_params_{0}.pkl'.format(monkey)),
                  'wb') as fid:
            pickle.dump(clf.best_params_, fid, -1)
        with open(path.join(data.BASEDIR, 'transcriber_rand_results_{0}.pkl'.format(monkey)),
                  'wb') as fid:
            pickle.dump((y_test, y_pred, labels))
        print monkey
        print classification_report(y_test, y_pred, target_names=labels)
