#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: svc_param_grid.py
# date: Wed August 06 01:26 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""svc_param_grid: parameter grid for svc grid search

"""
import numpy as np

param_grid = [{'kernel': ['rbf'],
               'gamma': np.logspace(-5, -2, 4),
               'C': np.logspace(0, 3, 50)},
              {'kernel': ['linear'],
               'C': np.logspace(0, 3, 50)}]

# small one for testing
# param_grid = [{'kernel': ['rbf'],
#                'gamma': [1e-4],
#                'C': [1]}]
