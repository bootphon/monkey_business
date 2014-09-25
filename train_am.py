#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: transcriber_state_transition.py
# date: Fri September 19 09:18 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""transcriber_state_transition: train and run transcription over entire audio
files using label state transitions to smooth out the acoustic posteriors

"""

from __future__ import division
import datetime
import os
import os.path as path
import cPickle as pickle


import numpy as np
import scipy.signal
import scipy.special
import scipy.stats



from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import RandomizedSearchCV
from sklearn.svm import SVC


import data
from resampling import Resampler

from am_loaders import VADLoader, AudioLoader


UNDERSAMPLING_RATIO = 1.1


def _load_files_labels(monkey, under_sample=True, under_sampling_ratio=1.0,
                       filelist=None, shuffle=True, frate=100):
    """Load datasets as X,y format

    Arguments:
    :param monkey: name of monkey, must be one of 'Blue_Fuller', 'Blue_Murphy',
      'Blue_merged', 'colobus', 'Titi_monkeys', 'all'
    :param under_sample: whether to uniformly undersample the NOISE class
    :param under_sampling_ratio: ratio of secondary class to shrink NOISE
      class to
    :param filelist: list of (dirname, fname) pairs. only datapoints in files in
      this list are used
    :param shuffle: whether to shuffle the dataset or not
    :param frate: framerate that will be used in feature extraction
    """
    if monkey == 'Blue_Fuller':
        dirname = 'Blue_monkeys_Fuller'
        annot = data.reduced_annotation(dirname, include_noise=True)
        if filelist:
           annot = {k: v for k, v in annot.iteritems()
                    if (dirname, k) in filelist}
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
        y = np.array(y, dtype=np.uint8)
        labels = labelset_remapped
    elif monkey == 'Blue_merged':
        dirname_murphy = 'Blue_monkeys'
        dirname_fuller = 'Blue_monkeys_Fuller'
        annot_murphy = data.reduced_annotation(dirname_murphy,
                                               include_noise=True)
        if filelist:
            annot_murphy = {k: v for k, v in annot_murphy.iteritems()
                            if (dirname_murphy, k) in filelist}
        annot_fuller = data.reduced_annotation(dirname_fuller,
                                               include_noise=True)
        if filelist:
            annot_fuller = {k: v for k, v in annot_fuller.iteritems()
                            if (dirname_fuller, k) in filelist}
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
        y = np.array(y, dtype=np.uint8)
        labels = labelset_murphy
    elif monkey == 'all':
        X = None
        y = None
        labels = ['NOISE']
        label2idx = None
        for m in ['colobus', 'Titi_monkeys', 'Blue_merged']:
            X_, y_, labels_ = _load_files_labels(m, shuffle=False,
                                                 filelist=filelist,
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
                                           len(labels) + len(labels_clean_))))
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
        # print 'len(annot.keys()) before filelist filter', len(annot.keys())
        # print annot.keys()
        # print monkey
        # print filelist
        if filelist:
            annot = {k: v for k, v in annot.iteritems()
                     if (monkey, k) in filelist}
        # print 'len(annot.keys()) after filelist filter', len(annot.keys())
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
        y = np.array(y, dtype=np.uint8)
    if under_sample:
        X, y = Resampler(ratio=under_sampling_ratio).resample(X, y)
    if shuffle:
        inds = np.random.permutation(X.shape[0])
        X = X[inds]
        y = y[inds]
    return X, y, labels


def _load_merged_annot(monkey):
    """Load train_test filenames from annot dicts,
    potentially with the marks merged if applicable"""
    if monkey == 'Blue_Fuller':
        remap = {'A': 'p', 'PY': 'p', 'KA': 'h', 'KATR': 'h', 'NOISE': 'NOISE'}
        annot = {('Blue_monkeys_Fuller', k): [data.Fragment(f.filename, f.interval, remap[f.mark])
                               for f in v]
                 for k, v in data.reduced_annotation('Blue_monkeys_Fuller',
                                                     include_noise=False)
                 .iteritems()}

    elif monkey == 'Blue_merged':
        # take murphy as the base
        annot = {('Blue_monkeys', k): v
                 for k, v in data.reduced_annotation('Blue_monkeys',
                                                     include_noise=False)
                 .iteritems()}

        # remap fuller onto it
        remap = {'A': 'p', 'PY': 'p', 'KA': 'h', 'KATR': 'h', 'NOISE': 'NOISE'}
        for filename, fragments in data.reduced_annotation('Blue_monkeys_Fuller',
                                                           include_noise=False).iteritems():
            annot[('Blue_monkeys_Fuller', filename)] = [data.Fragment(f.filename,
                                                         f.interval,
                                                         remap[f.mark])
                                           for f in fragments]
    elif monkey == 'all':
        annot = {}
        for m in ['colobus', 'Titi_monkeys', 'Blue_merged']:
            for filemarker, fragments in _load_merged_annot(m).iteritems():
                annot[filemarker] = [data.Fragment(f.filename,
                                                   f.interval,
                                                   (m, f.mark))
                                     for f in fragments]
    else:
        if monkey == 'Blue_Murphy':
            monkey = 'Blue_monkeys'
        annot = {(monkey, k): v
                 for k, v in data.reduced_annotation(monkey,
                                                     include_noise=False)
                 .iteritems()}
    return annot


def gen_train_test_files():
    """Generate train/test filesets"""
    ttdir = path.join(data.BASEDIR, 'traintestfiles')
    for monkey in ['Titi_monkeys', 'colobus', 'Blue_Fuller', 'Blue_Murphy',
                   'Blue_merged']:
        annot = _load_merged_annot(monkey)
        train, test = data.train_test_split_files(annot, test_size=0.2)
        with open(path.join(ttdir,
                            '{}_train_files.txt'.format(monkey)), 'w') as fid:
            for m, filename in train:
                fid.write('{0}\t{1}\n'.format(m, filename))
        with open(path.join(ttdir,
                            '{}_test_files.txt'.format(monkey)), 'w') as fid:
            for m, filename in test:
                fid.write('{0}\t{1}\n'.format(m, filename))


def get_train_test_files(monkey):
    ttdir = path.join(data.BASEDIR, 'traintestfiles')
    if monkey == 'all':
        train = []
        test = []
        for monkey in ['Titi_monkeys', 'colobus', 'Blue_merged']:
            tr, te = get_train_test_files(monkey)
            train += tr
            test += te
    else:
        train = []
        test = []
        for line in open(path.join(ttdir,
                                   '{}_train_files.txt'.format(monkey)), 'r'):
            train.append(tuple(line.strip().split('\t')))
        for line in open(path.join(ttdir,
                                   '{}_test_files.txt'.format(monkey)), 'r'):
            test.append(tuple(line.strip().split('\t')))
    return train, test

def nowstr():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    ttdir = path.join(data.BASEDIR, 'traintestfiles')
    try:
        os.makedirs(ttdir)
    except OSError:
        pass

    print nowstr(), 'generating train/test split'
    gen_train_test_files()
    outdir = path.join(data.BASEDIR, 'acoustic_models')
    try:
        os.makedirs(outdir)
    except OSError:
        pass
    for monkey in ['Titi_monkeys', 'colobus', 'Blue_Fuller', 'Blue_Murphy',
                   'Blue_merged', 'all']:
        print '-'*20
        print nowstr(), monkey
        print '-'*20
        train, _ = get_train_test_files(monkey)
        X_train, y_train, labels = _load_files_labels(monkey, filelist=train,
                                          under_sample=True,
                                          under_sampling_ratio=UNDERSAMPLING_RATIO)
        acoustic_pipeline = Pipeline([('data', FeatureUnion([('audio', AudioLoader()),
                                                             ('vad', VADLoader())])),
                                      ('svm', SVC(kernel='rbf', gamma=1e-5, C=20))])
        paramdist = {'svm__C': np.logspace(0, 4, 100),
                     'data__vad__stacksize': scipy.stats.randint(11, 51),
                     'data__vad__cut': [True, False],
                     'data__audio__stacksize': scipy.stats.randint(11, 51),
                    }

        acoustic_clf = RandomizedSearchCV(acoustic_pipeline, paramdist, n_iter=1000,
                                          verbose=1,
                                          cv=3, n_jobs=40)
        # acoustic_clf = RandomizedSearchCV(acoustic_pipeline, paramdist, n_iter=1,
        #                                   verbose=1,
        #                                   cv=2, n_jobs=1)
        acoustic_clf.fit(X_train, y_train)

        with open(path.join(outdir, 'am_params_{}.pkl'.format(monkey)), 'wb') as fid:
            pickle.dump(acoustic_clf.best_params_, fid, -1)
        with open(path.join(outdir, 'am_clf_{}.pkl'.format(monkey)), 'wb') as fid:
            pickle.dump(acoustic_clf.best_estimator_, fid, -1)
