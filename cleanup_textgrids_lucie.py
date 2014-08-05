#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: cleanup_textgrids_lucie.py
# date: Mon August 04 18:07 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""cleanup_textgrids_lucie: cleanup the textgrids Lucie prepared for James
Fuller's recordings. Remove all X and # intervals and insert '' intervals
between the remaining intervals.

"""

from __future__ import division
import os
import os.path as path

import data
from textgrid import TextGrid, Interval, Tier

indir = path.join(data.BASEDIR, 'Blue_monkeys_Fuller', 'textgrids_Lucie')
outdir = path.join(data.BASEDIR, 'Blue_monkeys_Fuller', 'textgrids')
try:
    os.makedirs(outdir)
except OSError:
    pass

for tgfile in sorted(list(data.rglob(indir, '*.TextGrid'))):
    basename = path.splitext(path.basename(tgfile))[0]
    print basename
    tg = TextGrid.read(tgfile)
    try:
        tier = tg['CALLS']
    except KeyError as e:
        print 'KeyError in', basename
        raise e
    intervals = [i for i in tg['CALLS'] if not i.mark in ['#', 'X']]
    start, end = tg.start, tg.end

    prev_end = start
    new_intervals = []
    for interval in intervals:
        new_intervals.append(Interval(prev_end, interval.start, ''))
        new_intervals.append(interval)
        prev_end = interval.end
    new_intervals.append(Interval(prev_end, end, ''))

    new_tier = Tier('', tg['CALLS'].start, tg['CALLS'].end,
                    'Interval', intervals)
    new_tg = TextGrid(start, end, [new_tier])

    outfile = path.join(outdir, basename + '.TextGrid')
    with open(outfile, 'w') as fid:
        new_tg.write(fid, fmt='long')
