#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: do_mad.py
# date: Thu July 10 01:49 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""do_mad:

"""

from __future__ import division

from collections import defaultdict
import os
import os.path as path
import fnmatch
import cPickle as pickle
import numpy as np

from sklearn.metrics import roc_curve, auc

from textgrid import TextGrid
from joblib import delayed, Parallel

import data
import mad

FRATE = 100
WINSIZE = 0.025

def rglob(rootdir, pattern):
    for root, _, files in os.walk(rootdir):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield path.join(root, basename)


def calc_activation_single(monkey, wavfile, vad, highpass=None):
    sig, fs = data.wavread(wavfile, highpass=highpass)
    return (monkey,
            path.splitext(path.basename(wavfile))[0],
            vad.calc_Lambda(sig, fs))

def calc_activation(n_cores=20, highpass=None):
    p = Parallel(n_jobs=n_cores,
                 verbose=0)(delayed(calc_activation_single)
                             (monkey, w, mad.MAD(epsilon=1e-4, NFFT=512,
                                                 win_size_sec=WINSIZE,
                                                 win_hop_sec=1/FRATE),
                              highpass)
                             for monkey in MONKEYS
                             for w in sorted(list(rglob(path.join(data.BASEDIR,
                                                                  monkey),
                                                        '*.wav'))))
    return p

def get_exp_call_activity(monkey, winsize_sec, hopsize_sec,
                          verbose=False):

    monkeydir = path.join(data.BASEDIR, monkey)
    calls = {}
    if verbose:
        total = 0
        total_call = 0
    for tgfile in rglob(path.join(monkeydir, 'textgrids'), '*.TextGrid'):
        basename = path.splitext(path.basename(tgfile))[0]
        try:
            wavfile = path.join(monkeydir, 'audio', basename + '.wav')
            sig, fs = data.wavread(wavfile)
        except IOError:
            continue

        winsize_smp = int(fs * winsize_sec)
        winhop_smp = int(fs * hopsize_sec)

        n_frames = int(np.floor((len(sig) - winsize_smp) / winhop_smp))

        if verbose:
            total_this = 0
            total_call_this = 0
            print basename, tgfile

        tg = TextGrid.read(tgfile)
        tier = tg[0]
        dec = np.zeros(n_frames, np.uint8)
        for interval in tier:
            if verbose:
                length = interval.end - interval.start
                total += length
                total_this += length
            mark = interval.mark.strip().replace(' ', '')
            if mark != '':  # so we have a call here
                if verbose:
                    print ('    mark: {0} length: {1:.2f}s [{2:.3f}, {3:.3f}]'
                           .format(mark, length,
                                   interval.start, interval.end))
                    total_call += length
                    total_call_this += length
                ival_start_smp = int(fs * interval.start)
                ival_stop_smp = int(fs * interval.end)
                frame_start = int(ival_start_smp / winhop_smp)
                frame_end = int(ival_stop_smp / winhop_smp)
                dec[frame_start: frame_end] = 1
        calls[basename] = dec
        if verbose:
            print '  call proportion: {0:.2f}s / {1:.2f}s [{2:.2f}%]'.format(
                total_call_this, total_this, total_call_this * 100 / total_this)
    if verbose:
        print 'call proportion: {0:.3f}s / {1:.3f}s [{2:.2f}%]'.format(
            total_call, total, total_call * 100 / total)
    return calls

MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus', 'Blue_monkeys_Fuller']
NCORES = 40

if __name__ == '__main__':
    best_auc = 0
    best_r = None
    best_hp = None
    highpasses = np.arange(0, 4000, 100)
    exp_calls_all = {monkey: get_exp_call_activity(monkey, WINSIZE, 1/FRATE)
                     for monkey in MONKEYS}
    for highpass in highpasses:
        p = calc_activation(n_cores=NCORES, highpass=highpass)
        r = defaultdict(dict)
        for monkey, fname, arr in p:
            r[monkey][fname] = arr

        auc_score = 0
        for monkey in r:
            exp_calls = exp_calls_all[monkey]
            keys = sorted(list(set(exp_calls.keys()).intersection(set(r[monkey].keys()))))
            activation = np.hstack((r[monkey][k] for k in keys))
            exp_calls = np.hstack((exp_calls[k] for k in keys))

            fpr, tpr, _ = roc_curve(exp_calls, activation)
            auc_score += auc(fpr, tpr)
        auc_score /= 4
        print highpass, auc_score
        if auc_score > best_auc:
            best_auc = auc_score
            best_r = r
            best_hp = highpass
    print 'BEST:'
    print best_auc
    print best_hp

    for monkey in best_r:
        with open(path.join(data.BASEDIR, 'mad_{0}.pkl'.format(monkey)), 'wb') as fid:
            pickle.dump(best_r[monkey], fid, -1)
