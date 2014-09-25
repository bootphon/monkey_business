#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: resampling.py
# date: Mon August 25 18:43 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""resampling:

"""

from __future__ import division

import numpy as np

class Resampler(object):
    def __init__(self, ratio=1.0, method='uniform'):
        assert(method in ['uniform', 'tomek'])
        if method == 'tomek':
            raise NotImplementedError('Tomek undersampling is not implemented.'
                                      ' Use "uniform" instead.')
        self.method = method
        self.ratio = ratio

    def resample(self, X, y):
        labels = np.unique(y)
        labelcounts = np.bincount(y)

        maj_class = np.argmax(labelcounts)
        sec_class = np.argsort(labelcounts)[-2]

        nsamples = int(labelcounts[sec_class] * self.ratio)
        maj_samples = np.random.choice(np.nonzero(y==maj_class)[0],
                                       size=nsamples,
                                       replace=False)
        inds = np.hstack((np.nonzero(y==label)[0]
                          if label != maj_class else maj_samples)
                         for label in labels)
        return X[inds], y[inds]
