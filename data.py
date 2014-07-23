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
from itertools import imap
import warnings
from functools import Counter
from operator import add

import numpy as np
from scikits.audiolab import wavread

import spectral
from textgrid import TextGrid

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
    # do some filtering to get rid of weird labels in titi monkeys
    counts = reduce(add, (Counter(f.mark for f in annot[fname])
                          for fname in annot))
    annot = {k: [f for f in v if counts[f.mark] >= min_samples]
             for k, v in annot.iteritems()}
    labelset = sorted(k for k in counts if counts[k] >= min_samples)
    label2idx = dict(zip(labelset, range(len(labelset))))
    nsamples = sum(imap(len, annot.itervalues()))

    frate = 100
    encoder = spectral.Spectral(nfilt=nfilt, fs=16000, wlen=0.025, frate=frate,
                                compression='log', nfft=1024, do_dct=False,
                                do_deltas=False, do_deltasdeltas=False)

    X = np.empty((nsamples, nframes*nfilt), dtype=np.double)
    y = np.empty((nsamples,), dtype=np.uint8)
    idx = 0
    for fname in annot:
        spec = load_wav(path.join(BASEDIR, monkey, 'audio', fname + '.wav'),
                        encoder)
        for fragment in annot[fname]:
            start_frame = int(fragment.interval.start * frate)
            if start_frame + nframes >= spec.shape[0]:
                s = spec[start_frame: start_frame + nframes]
                extra_zeros = np.zeros((start_frame + nframes - spec.shape[0],
                                        nfilt),
                                       dtype=s.dtype)
                x = np.vstack((s, extra_zeros))
                X[idx] = np.hstack(x)
            else:
                X[idx] = np.hstack(spec[start_frame: start_frame + nframes])
            y[idx] = label2idx[fragment.mark]
            idx += 1
    return X[:idx], y[:idx], labelset


def load_wav(wavfile, encoder):
    """Load audio from wave file and do spectral transform

    Arguments:
    :param wavfile: audio filename
    :param encoder: Spectral object

    :return nframes x nfilts array
    """
    sig, fs, _ = wavread(wavfile)
    if encoder.config['fs'] != fs:
        raise ValueError('sampling rate should be {0}, not {1}. '
                         'please resample.'.format(encoder.config['fs'], fs))
    if len(sig.shape) > 1:
        warnings.warn('stereo audio found, will merge channels')
        sig = (sig[:, 0] + sig[:, 1]) / 2
    return encoder.transform(sig)
