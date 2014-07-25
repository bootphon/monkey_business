#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: data.py
# date: Wed July 23 19:37 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""data: interface to corpus

"""

from __future__ import division
import os
import os.path as path
import fnmatch
from collections import namedtuple, defaultdict
from itertools import imap, chain, combinations
import warnings
from collections import Counter
from operator import add

import numpy as np
import scipy.signal
from scipy.io import wavfile as siowavfile

import spectral
from textgrid import TextGrid

BASEDIR = "/fhgfs/bootphon/scratch/gsynnaeve/monkey_sounds"

Interval = namedtuple('Interval', ['start', 'end'])
Fragment = namedtuple('Fragment', ['filename', 'interval', 'mark'])


def rglob(rootdir, pattern):
    for root, _, files in os.walk(rootdir):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield path.join(root, basename)


def get_annotation(monkey, include_noise=False):
    """Get the annotation for monkey

    :param include_noise: do not exclude noise intervals

    :return dict from filename to list of Fragments
    """
    monkeydir = path.join(BASEDIR, monkey)
    annot = defaultdict(list)
    for tgfile in rglob(path.join(monkeydir, 'textgrids'), '*.TextGrid'):
        filename = path.splitext(path.basename(tgfile))[0]
        if not path.exists(path.join(monkeydir, 'audio', filename + '.wav')):
            continue
        tg = TextGrid.read(tgfile)
        tier = tg.tiers[0]
        for interval in tier:
            mark = interval.mark.strip()
            if mark == '' and not include_noise:
                continue
            fragment = Fragment(filename,
                                Interval(interval.start - tier.start,
                                         interval.end - tier.start),
                                mark)
            annot[filename].append(fragment)
    return annot


def stack_up(spec, start_frame, nframes):
    nfilt = spec.shape[1]
    x = spec[start_frame: start_frame + nframes].flatten()
    if x.shape[0] < nframes * nfilt:
        x = np.pad(x, (0, nframes * nfilt - x.shape[0]), mode='constant')
    return x


def stack_all(a, nframes):
    return np.hstack(np.roll(a, -i, 0)
                     for i in xrange(nframes))[:a.shape[0] - nframes + 1]


def load_data_stacked_annot(monkey, annot, encoder, nframes, highpass=None):
    labelset = sorted(list(set(f.mark for fname in annot
                               for f in annot[fname])))
    label2idx = dict(zip(labelset, range(len(labelset))))
    nsamples = sum(imap(len, annot.itervalues()))

    frate = encoder.config['frate']
    nfilt = encoder.config['nfilt']
    X = np.empty((nsamples, nframes * nfilt),
                 dtype=np.double)
    y = np.empty((nsamples,), dtype=np.uint8)
    idx = 0
    for fname in annot:
        spec = load_wav(path.join(BASEDIR, monkey, 'audio', fname + '.wav'),
                        encoder, highpass=highpass)
        for fragment in annot[fname]:
            X[idx] = stack_up(spec, int(fragment.interval.start * frate),
                              nframes)
            y[idx] = label2idx[fragment.mark]
            idx += 1
    return X[:idx], y[:idx], labelset


def load_data_stacked(monkey, nframes=30, nfilt=40, include_noise=False,
                      min_samples=50):
    """Loads audio data for monkey as stacked

    Arguments:
    :param monkey: name of the monkey
    :param nframes: number of frames to stack
    :param nfilt: number of filterbanks
    :param include_noise: do not exclude noise intervals
    :param min_samples: minimum number of samples for a class to be used

    :return
      X: audio representation, ndarray (nsamples x nfilt * nframes)
      y: labels as int, ndarray (nsamples)
      labelset: list of call names (maps onto ints in y)
    """
    annot = get_annotation(monkey, include_noise=include_noise)
    counts = reduce(add, (Counter(f.mark for f in annot[fname])
                          for fname in annot))
    annot = {k: [f for f in v if counts[f.mark] >= min_samples]
             for k, v in annot.iteritems()}
    # labelset = sorted(list(set(f.mark for fname in annot for f in annot[fname])))
    # # labelset = sorted(k for k in counts if counts[k] >= min_samples)
    # label2idx = dict(zip(labelset, range(len(labelset))))
    # nsamples = sum(imap(len, annot.itervalues()))

    frate = 100
    encoder = spectral.Spectral(nfilt=nfilt, fs=16000, wlen=0.025, frate=frate,
                                compression='log', nfft=1024, do_dct=False,
                                do_deltas=False, do_deltasdeltas=False)

    X, y, labelset = load_data_stacked_annot(monkey, annot, encoder, nframes)
    return X, y, labelset


def load_wav(wavfile, encoder, highpass=None):
    """Load audio from wave file and do spectral transform

    Arguments:
    :param wavfile: audio filename
    :param encoder: Spectral object
    :param highpass: filter first at frequency, if None don't filter

    :return nframes x nfilts array
    """
    fs, sig = siowavfile.read(wavfile)
    if not highpass is None:
        cutoff = highpass / (0.5 * fs)
        b, a = scipy.signal.butter(5, cutoff, btype='highpass')
        sig = scipy.signal.lfilter(b, a, sig)

    if encoder.config['fs'] != fs:
        raise ValueError('sampling rate should be {0}, not {1}. '
                         'please resample.'.format(encoder.config['fs'], fs))
    if len(sig.shape) > 1:
        warnings.warn('stereo audio found, will merge channels')
        sig = (sig[:, 0] + sig[:, 1]) / 2
    return encoder.transform(sig)


def train_test_split_files(annot, test_size=0.2):
    """Split files in annotation in training and test set,
    preserving distribution of labels

    :param annot: dict from filename to list of fragments
    :param test_size: proportion of test wrt train set

    :return
      train: list of filenames
      test: list of filenames
    """
    filenames  = sorted(annot.keys())
    counts = [Counter(f.mark for f in annot[fname])
              for fname in filenames]
    total = reduce(add, counts)
    target = {k: int(total[k] * (1-test_size)) for k in total}

    # BRUUUUUUTEFOOOOOORCE!!!!
    cutoff = 20
    mincost = np.inf
    bestsol = None
    for indices in chain.from_iterable(combinations(xrange(len(counts)), k)
                                       for k in xrange(1, len(counts)-1)):
        counter = reduce(add, [counts[i] for i in indices])
        cost = sum(abs(target[k] - counter[k]) for k in target)
        if cost < mincost:
            mincost = cost
            bestsol = indices
        if mincost < cutoff:
            break

    bestsol = set(bestsol)

    train = [filenames[i] for i in bestsol]
    test = [f for idx, f in enumerate(filenames) if not idx in bestsol]
    return train, test
