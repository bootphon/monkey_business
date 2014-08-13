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
from collections import namedtuple, defaultdict, Counter
from itertools import imap, chain, combinations
import warnings
from operator import add
import cPickle as pickle

import numpy as np
import scipy.signal
from scipy.io import wavfile as siowavfile

import spectral
from textgrid import TextGrid

# BASEDIR = "/fhgfs/bootphon/scratch/gsynnaeve/monkey_sounds"

BASEDIR = path.join(os.environ['HOME'], 'data', 'monkey_sounds')


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
            print 'missing audio file:', monkey, filename + '.wav'
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


def reduced_annotation(monkey, min_samples=50, include_noise=True):
    """Load annotation for monkey and replace labels that occur less than
    min_samples times with 'NOISE', the noise_1 label

    :param monkey: monkey
    :param min_samples: labels with less than this amount are relabelled as noise_1

    :return annot: dict from filename to list of Fragments
    """
    annot = get_annotation(monkey, include_noise=True)
    counts = reduce(add, (Counter(f.mark for f in annot[fname])
                          for fname in annot))
    annot = {k: [((f
                   if f.mark != ''
                   else Fragment(f.filename,
                                      f.interval,
                                      'NOISE'))
                  if counts[f.mark] >= min_samples
                  else Fragment(f.filename,
                                     f.interval,
                                     'NOISE')) for f in v]
             for k, v in annot.iteritems()}
    if not include_noise:
        annot = {k: [f for f in v if f.mark != 'NOISE']
                 for k, v in annot.iteritems()}
    return annot


def stack_from_frame(spec, start_frame, nframes):
    nfilt = spec.shape[1]
    x = spec[start_frame: start_frame + nframes].flatten()
    if x.shape[0] < nframes * nfilt:
        x = np.pad(x, (0, nframes * nfilt - x.shape[0]), mode='constant')
    return x


def stack_array(arr, nframes):
    return np.hstack(np.roll(arr, -i, 0)
                     for i in xrange(nframes))[:arr.shape[0] - nframes + 1]


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
            X[idx] = stack_from_frame(spec, int(fragment.interval.start * frate),
                              nframes)
            y[idx] = label2idx[fragment.mark]
            idx += 1
    return X[:idx], y[:idx], labelset


def load_data_stacked(monkey, nframes=30, nfilt=40, include_noise=False,
                      min_samples=50):
    """Loads audio data for monkey as stacked. Only intervals.

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
    sig, fs = wavread(wavfile, highpass=highpass)

    if encoder.config['fs'] != fs:
        raise ValueError('sampling rate should be {0}, not {1}. '
                         'please resample.'.format(encoder.config['fs'], fs))
    if len(sig.shape) > 1:
        warnings.warn('stereo audio found, will merge channels')
        sig = (sig[:, 0] + sig[:, 1]) / 2
    return encoder.transform(sig)


def wavread(wavfile, highpass=None):
    fs, sig = siowavfile.read(wavfile)
    if not highpass is None:
        cutoff = highpass / (0.5 * fs)
        b, a = scipy.signal.butter(5, cutoff, btype='highpass')
        sig = scipy.signal.lfilter(b, a, sig)
    return sig, fs


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
    # target = {k: int(total[k] * (1-test_size)) for k in total}

    target = {k: int(total[k] * test_size) for k in total}

    print ' ', target

    # BRUUUUUUTEFOOOOOORCE!!!!
    cutoff = sum(target.itervalues()) * 0.05
    # print ' ', cutoff
    mincost = np.inf
    bestsol = None
    length_prev = -1
    all_too_large = False
    for idx, indices in enumerate(chain.from_iterable(combinations(xrange(len(counts)), k)
                                                      for k in xrange(1, len(counts)-1))):
        length = len(indices)
        if length > length_prev:
            if all_too_large:
                break
            else:
                length_prev = length
                all_too_large = True

        counter = reduce(add, [counts[i] for i in indices])
        cost = sum(abs(target[k] - counter[k]) for k in target)
        all_too_large = all_too_large and all(counter[k] > target[k] for k in target)
        if cost < mincost:
            mincost = cost
            bestsol = indices
            # print ' ', idx, mincost
        if mincost < cutoff:
            break

    bestsol = set(bestsol)

    train = [f for idx, f in enumerate(filenames) if not idx in bestsol]
    test = [filenames[i] for i in bestsol]
    return train, test


def load_data_full_stacks(monkey, nfilt=40, stacksize=30, highpass=2000, min_samples=50):
    """

    Arguments:
    :param monkey:
    :param nfilt:
    :param stacksize:
    :param highpass:
    :param min_samples:
    """
    annot = reduced_annotation(monkey, min_samples=min_samples)

    X_train = {}
    X_test = {}
    y_train = {}
    y_test = {}

    frate = 100
    encoder = spectral.Spectral(nfilt=nfilt, fs=16000, wlen=0.025, frate=frate,
                                compression='log', nfft=1024, do_dct=False,
                                do_deltas=False, do_deltasdeltas=False)

    train_files, test_files = train_test_split_files(annot)

    labelset = sorted(list(set((f.mark if f.mark != '' else 'NOISE')
                               for fname in annot
                               for f in annot[fname])) + ['NOISE_ACT'])

    with open(path.join(BASEDIR, 'pred_lambdas_{0}.pkl'.format(monkey)),
              'rb') as fid:
        pred_lambda = pickle.load(fid)

    act_intervals = {}
    for fname in pred_lambda:
        act_intervals[fname] = speech_activity_to_intervals(pred_lambda[fname],
                                                            threshold=0.5,
                                                            winhop=0.025)

    annot_train = {fname: annot[fname] for fname in train_files}
    for fname in annot_train:
        X, y = load_Xy(monkey, fname, encoder, annot_train[fname],
                       act_intervals[fname], labelset,
                       frate, highpass, stacksize)
        X_train[fname] = X
        y_train[fname] = y

    annot_test = {fname: annot[fname] for fname in test_files}
    for fname in annot_test:
        X, y = load_Xy(monkey, fname, encoder, annot_test[fname],
                       act_intervals[fname], labelset,
                       frate, highpass, stacksize)
        X_test[fname] = X
        y_test[fname] = y

    return X_train, X_test, y_train, y_test, labelset


def load_Xy(monkey, fname, encoder, manual_fragments, auto_intervals, labelset,
            frate=100, highpass=2000, stacksize=30):
    """Loads audio (X) and labels (y) for a specific monkey and filename.
    The labels are constructed from manual fragment annotation and output
    from the voice activity detector. Audio is stacked. Labels reference majority
    labels over stacked frames.

    Arguments:
    :param monkey:
    :param fname:
    :param encoder:
    :param manual_fragments:
    :param auto_intervals:
    :param frate:
    :param labelset:
    :param highpass:

    """
    wavfile = path.join(BASEDIR, monkey, 'audio', fname + '.wav')
    spec = load_wav(wavfile, encoder, highpass=highpass)
    X = stack_array(spec, stacksize)
    label2idx = dict(zip(labelset, range(len(labelset))))

    # annotate the stacks
    y = np.zeros(spec.shape[0], dtype=np.uint8)

    # first by manual annotation
    for fragment in manual_fragments:
        interval = fragment.interval
        start_frame = int(interval.start * frate)
        end_frame = int(interval.end * frate)
        y[start_frame: end_frame] = label2idx[fragment.mark]

    # then by output from vad
    for interval in auto_intervals:
        start_frame = int(interval.start * frate)
        end_frame = int(interval.end * frate)
        mark = label2idx['NOISE_ACT']
        mask = np.zeros(y.shape, dtype=np.bool)
        mask[start_frame: end_frame] = True
        y[np.logical_and(mask, y==label2idx['NOISE'])] = mark
    # y = np.hstack((y, np.ones(stacksize, dtype=np.uint8) * label2idx['NOISE']))
    # y = np.hstack(np.roll(y[:, np.newaxis], -i, 0)
    #               for i in xrange(stacksize))[:y.shape[0] - stacksize + 1]
    # y = scipy.stats.mode(y, 1)[0].astype(int).flatten()

    y = y[:X.shape[0]]
    if X.shape[0] != y.shape[0]:
        print 'X.shape', X.shape
        print 'y.shape', y.shape

    assert(X.shape[0] == y.shape[0])
    return X, y



def speech_activity_to_intervals(pred_lambda, threshold=0.5, winhop=0.025):
    """Return list of intervals where speech activity is detected

    Arguments:
    :param pred_lambda: activation output from vad
    :param threshold:
    """
    d = (pred_lambda > threshold).astype(int)
    diff = np.hstack(([1], np.diff(d)))
    idx = np.where(diff)[0]
    groups = np.diff(np.hstack((idx, [len(d)])))
    pos_idx = idx[np.nonzero(d[idx])[0]]
    pos_lens = groups[np.nonzero(d[idx])[0]]
    return [Interval(winhop*s, winhop*e)
            for s, e in zip(pos_idx, pos_idx + pos_lens)]
