#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: am_loaders.py
# date: Thu September 25 16:51 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""am_loaders:

"""

from __future__ import division

# stdlib
import warnings
import os.path as path
import cPickle as pickle

# external
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from scikits.audiolab import wavread
import scipy.signal

# own
import spectral

# project
import data


# thresholds for VAD cutoff
THETAS = {'Blue_monkeys'        : 0.5066,
          'Blue_monkeys_Fuller' : 0.5062,
          'Titi_monkeys'        : 0.5007,
          'colobus'             : 0.5060}
FRATE  = 100


def hpfilter(sig, fs, cutoff, order=5):
    cutoff = cutoff / (0.5 * fs)
    b, a = scipy.signal.butter(order, cutoff, btype='highpass')
    return scipy.signal.lfilter(b, a, sig)


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
def _load_vad(monkey, cut=False):
    theta = THETAS[monkey]
    key = monkey
    if not key in _vad_cache:
        with open(path.join(data.BASEDIR, 'mad_{0}.pkl'.format(monkey))) as fid:
            pred_vad = pickle.load(fid)
        for fname in pred_vad:
            if cut:
                pred_vad[fname] = scipy.special.expit(pred_vad[fname]) >= theta
            else:
                pred_vad[fname] = scipy.special.expit(pred_vad[fname])
        _vad_cache[key] = pred_vad
    return _vad_cache[key]

class AudioLoader(TransformerMixin, BaseEstimator):
    def __init__(self, nfilt=40, highpass=None, normalize=True, stacksize=31):
        self.nfilt = nfilt
        self.highpass = highpass
        self.normalize = normalize
        self.stacksize = stacksize

    def transform(self, X, y=None, **transform_params):
        specs = {}
        for monkey, fname in set(zip(X[:,0], X[:,1])):
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
    def __init__(self, normalize=True, stacksize=31, cut=True):
        self.normalize = normalize
        self.stacksize = stacksize
        self.cut = cut

    def transform(self, X, y=None, **transform_params):
        vads = {}
        for monkey in np.unique(X[:, 0]):
            vads[monkey] = _load_vad(monkey, self.cut)
        nsamples = X.shape[0]
        X_ = np.empty((nsamples, self.stacksize), dtype=np.double)
        for idx, (monkey, fname, frameno, _, _) in enumerate(X):
            try:
                X_[idx] = vads[monkey][fname][int(frameno)]
            except KeyError:
                # this is here because the filenames changed after the voice
                # activity was already extracted
                X_[idx] = vads[monkey][fname.replace('_cut', '')][int(frameno)]
        if self.normalize:
            X_ = (X_ - X_.mean(0)) / X_.std(0)
        return X_

    def fit(self, X, y=None, **fit_params):
        return self
