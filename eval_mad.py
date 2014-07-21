#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: eval_mad.py
# date: Wed July 09 22:22 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""eval_mad:

"""

from __future__ import division
import os
import os.path as path
import fnmatch
import cPickle as pickle

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scikits.audiolab import wavread

from textgrid import TextGrid

BASEDIR = '/home/mwv/data/monkey_sounds/'
MONKEYS = ['Titi_monkeys', 'Blue_monkeys', 'colobus']
# MONKEYS = ['Titi_monkeys']


def rglob(rootdir, pattern):
    for root, _, files in os.walk(rootdir):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                yield path.join(root, basename)


def find_closest(arr, val):
    if arr.shape[0] == 0:
        return -1
    if arr.shape[0] == 1:
        return arr[0]
    idx = np.searchsorted(arr, val)
    idx = np.clip(idx, 1, arr.shape[0] - 1)
    left = arr[idx - 1]
    try:
        right = arr[idx]
    except IndexError as e:
        print idx, arr.shape
        raise e
    idx -= val - left < right - val
    return idx


def prec_rec_fscore(expected_starts, predicted_starts, tol=0.5):
    predicted_arr = np.array(predicted_starts)
    exp_pred = 0
    for exp in sorted(expected_starts):
        closest = find_closest(predicted_arr, exp)
        if abs(closest - exp) < tol:
            exp_pred += 1

    npred = len(predicted_arr)

    if npred == 0:
        prec = 0
    else:
        prec = exp_pred / npred
    nexp = len(expected_starts)
    if nexp == 0:
        rec = 0
    else:
        rec = exp_pred / nexp
    if rec == 0 and prec == 0:
        fscore = 0
    else:
        fscore = 2 * prec * rec / (prec + rec)
    return prec, rec, fscore


def get_exp_starts(monkey):
    starts = {}
    for tgfile in rglob(path.join(BASEDIR, monkey, 'textgrids'), '*.TextGrid'):
        basename = path.splitext(path.basename(tgfile))[0]
        tg = TextGrid.read(tgfile)
        if 'calls' in tg:
            tier = tg['calls']
        else:
            tier = tg[0]
        starts[basename] = [x.start for x in tier
                                   if x.mark.strip() != '']
    return starts

def get_exp_call_activity(monkey, winsize_sec, hopsize_sec,
                          verbose=False):

    monkeydir = path.join(BASEDIR, monkey)
    calls = {}
    if verbose:
        total = 0
        total_call = 0
    for tgfile in rglob(path.join(monkeydir, 'textgrids'), '*.TextGrid'):
        basename = path.splitext(path.basename(tgfile))[0]
        try:
            wavfile = path.join(monkeydir, 'audio', basename + '.wav')
            sig, fs, _ = wavread(wavfile)
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

def get_lambda_est(monkey):
    with open('/home/mwv/data/monkey_sounds/pred_lambdas_{0}.pkl'.format(monkey), 'rb') as fid:
        p = pickle.load(fid)
    return {k.replace(' ', '_'): expit(v) for k, v in p.iteritems()}

def lambda_to_starts(lambda_est, threshold, winhop=0.025):
    """Return time in sec where lambda_est starts to go above threshold
    """
    l_th = lambda_est > threshold
    return np.where(((np.roll(l_th, 1) - l_th) != 0).astype(int))[0][::2] * winhop

def get_pred_starts(lambda_ests, threshold):
    return {k: lambda_to_starts(v, threshold)
            for k, v in lambda_ests.iteritems()}

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn
    tprs = {}
    fprs = {}
    precs = {}
    recs = {}
    aucrocs = {}
    aucprcs = {}
    for monkey in MONKEYS:
        print monkey
        pred_lambda = get_lambda_est(monkey)
        exp_calls = get_exp_call_activity(monkey, 0.05, 0.025)
        keys = sorted(list(set(exp_calls.keys()).intersection(set(pred_lambda.keys()))))
        for key in keys:
            if pred_lambda[key].shape != exp_calls[key].shape:
                print ('wrong number of frames for {0}. pred_lambda: {1} exp_calls: {2}'
                       .format(key, pred_lambda[key].shape, exp_calls[key].shape))
        pred_all = np.hstack((pred_lambda[k] for k in keys))
        exp_all = np.hstack((exp_calls[k] for k in keys))

        # ROC
        fpr, tpr, _ = roc_curve(exp_all, pred_all)
        fprs[monkey] = fpr
        tprs[monkey] = tpr
        auc_score = auc(fpr, tpr)
        aucrocs[monkey] = auc_score

        fig = plt.figure()
        plt.plot(fpr, tpr, label='AUC = {0:.3f}'.format(auc_score))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='best')
        plt.title('ROC curve for {0}'.format(monkey))
        plt.savefig('roc_{0}.png'.format(monkey))

        # PREC/REC curve
        prec, rec, _ = precision_recall_curve(exp_all, pred_all)
        precs[monkey] = prec
        recs[monkey] = rec
        auc_score = auc(prec, rec, reorder=True)
        aucprcs[monkey] = auc_score

        fig = plt.figure()
        plt.plot(rec, prec, label='AUC = {0:.3f}'.format(auc_score))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.legend(loc='best')
        plt.title('precision/recall curve for {0}'.format(monkey))
        plt.savefig('prc_{0}.png'.format(monkey))

    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for monkey in tprs:
        plt.plot(fprs[monkey],
                 tprs[monkey],
                 label='{0} (AUC={1:.3f})'.format(monkey, aucrocs[monkey]))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    plt.title('ROC curves')
    plt.savefig('roc_all.png')

    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for monkey in precs:
        plt.plot(recs[monkey],
                 precs[monkey],
                 label='{0} (AUC={1:.3f})'.format(monkey, aucprcs[monkey]))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc='best')
    plt.title('precision/recall curves')
    plt.savefig('prc_all.png')
