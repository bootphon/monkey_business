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

from __future__ import division
import scikits.audiolab as audiolab
import os
import os.path as path
import fnmatch
import cPickle as pickle

from joblib import delayed, Parallel

import mad

def rglob(rootdir, pattern):
    for root, _, files in os.walk(rootdir):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield path.join(root, basename)


def get_Lambda_single(wavfile, vad):
    sig, fs, _ = audiolab.wavread(wavfile)
    return path.splitext(path.basename(wavfile))[0], vad.calc_Lambda(sig, fs)


def pred_Lambdas(monkey, n_cores=20):
    wavfiles = sorted(list(rglob(path.join(BASEDIR, monkey), '*.wav')))

    p = Parallel(n_jobs=n_cores, verbose=11)(delayed(get_Lambda_single)
                                             (w, mad.MAD(epsilon=1e-4,
                                                         NFFT=512))
                                             for w in wavfiles)
    return dict(p)

BASEDIR = path.join(os.environ['HOME'], 'data', 'monkey_sounds')
MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus', 'gibbons']
# MONKEYS = ['Titi_monkeys', 'Blue_monkeys']
NCORES = 20

if __name__ == '__main__':
    for monkey in MONKEYS:
        print monkey
        pred_lambdas = pred_Lambdas(monkey, n_cores=NCORES)
        with open('pred_lambdas_{0}.pkl'.format(monkey), 'wb') as fid:
            pickle.dump(pred_lambdas, fid, -1)
