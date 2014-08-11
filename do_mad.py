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


def get_Lambda_single(monkey, wavfile, vad):
    sig, fs, _ = audiolab.wavread(wavfile)
    return monkey, path.splitext(path.basename(wavfile))[0], vad.calc_Lambda(sig, fs)

def pred_Lambdas_all(n_cores=20):
    p = Parallel(n_jobs=n_cores, verbose=11)(delayed(get_Lambda_single)
                   (monkey, w, mad.MAD(epsilon=1e-4, NFFT=512))
                   for monkey in MONKEYS
                   for w in sorted(list(rglob(path.join(BASEDIR, monkey),
                                              '*.wav'))))
    return p

BASEDIR = path.join(os.environ['HOME'], 'data', 'monkey_sounds')
MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus', 'Blue_monkeys_Fuller']
NCORES = 40

if __name__ == '__main__':
    p = pred_Lambdas_all(n_cores=NCORES)
    r = defaultdict(dict)
    for monkey, fname, arr in p:
        r[monkey][fname] = arr
    for monkey in r:
        with open(path.join(BASEDIR, 'mad_{0}.pkl'.format(monkey)), 'wb') as fid:
            pickle.dump(r[monkey], fid, -1)
