#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: call_detector.py
# date: Thu July 31 01:15 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""call_detector: simple energy based call detector

"""

from __future__ import division
import os.path as path

import numpy as np
from scipy.signal import butter, lfilter
import scipy.io.wavfile

from data import rglob
from textgrid import TextGrid, Interval, Tier


def wavread(fname):
    fs, sig = scipy.io.wavfile.read(fname)
    sig = sig / (max(abs(sig.min()), sig.max()))
    return sig, fs


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rms_energy(sig, N=5):
    return np.sqrt(np.sum(rolling_window(np.hstack((np.zeros(N//2),
                                                    sig,
                                                    np.zeros(N//2)))**2,
                                         N),
                          axis=1) / N)

def highpass(sig, fs, cut, order=5):
    cut = cut / (0.5*fs)
    b, a = butter(order, cut, btype='highpass')
    return lfilter(b, a, sig)


def detect_transients(sig, fs, winlen=0.05, threshold=0.005, minlen=0.01):
    winlen = int(fs * winlen)
    minlen = int(fs * minlen)
    cut = rms_energy(sig, N=winlen) > threshold
    transients = np.where(((np.roll(cut, 1) - cut) != 0).astype(int))[0]
    transients = transients.reshape((transients.shape[0]//2, 2))
    return transients[(transients[:,1] - transients[:,0]) > minlen]


def get_intervals(sig, fs, winlen=0.05, threshold=0.005, minlen=0.01):
    """Return a list of intervals (in sec) and an 'X' mark if they contain
    energy, otherwise a '#' mark

    Arguments:
    :param sig: s
    :param fs:
    :param winlen: windowlength for the rms in seconds
    :param threshold: energy cutoff
    :param minlen: minimum length of an interval in seconds
    """
    transients = detect_transients(sig, fs, winlen, threshold, minlen)
    intervals = []
    prev_end = 0.0
    if transients.shape[0] == 0:
        intervals.append(Interval(0.0, len(sig) / fs, '#'))
    else:
        for start, end in transients:
            intervals.append(Interval(prev_end, start / fs, '#'))
            intervals.append(Interval(start / fs, end / fs, 'X'))
            prev_end = end / fs
        intervals.append(Interval(prev_end, len(sig) / fs, '#'))
    return intervals


def make_textgrid(intervals):
    tier = Tier('CALLS', intervals[0].start, intervals[-1].end, 'Interval', intervals)
    tg = TextGrid(intervals[0].start, intervals[-1].end, [tier])
    return tg


if __name__ == '__main__':
    basedir = '/home/mwv/ownCloud/monkey_sounds/C. mitis male recordings (J. Fuller, 07-2014)/'
    indir = path.join(basedir, 'audio')
    outdir = path.join(basedir, 'textgrids')
    for wavfile in rglob(indir, '*.wav'):
        basename = path.splitext(path.basename(wavfile))[0]
        print basename
        sig, fs = wavread(path.join(indir, wavfile))
        sig = highpass(sig, fs, 2000)
        intervals = get_intervals(sig, fs)

        tg = make_textgrid(intervals)
        with open(path.join(outdir, basename + '.TextGrid'), 'w') as fid:
            tg.write(fid, fmt='long')
