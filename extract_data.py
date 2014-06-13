""" python extract_data.py
You need to set the DATASET_PATH and other constants. Produces a *.npz.
"""

from spectral import Mel  # you need https://github.com/mwv/spectral
import numpy as np
from scipy.io import wavfile
from collections import defaultdict, Counter
import glob

DATASET_PATH = "/Users/gabrielsynnaeve/ownCloud/Shared/Monkey Sounds/Blue Monkeys/*[WAV|wav]"
OUTPUT_FILE = "blue_monkeys"
GLOBAL_NORMALIZE = False  # global normalize => not at the file level
STACK = 20  # in frames
STRIDE = 10  # in frames
nfbanks = 40
wfbanks = 0.025 # 25ms
rfbanks = 100 # 10ms


def parse_textgrid(fname):
    with open(fname) as f:
        l = f.readlines()
    res = defaultdict(lambda: [])
    for i, line in enumerate(l):
        if 'text = ' in line:
            line = line.rstrip('\n')
            label = line.split('=')[1].strip().strip('"').rstrip('"').strip()
            if label != "":
                res[label].append((float(l[i-2].split('=')[1]),
                    float(l[i-1].split('=')[1])))
    return res


def vote_labels(l, length, stride, stack, omit_na_if_possible=False):
    res = np.ndarray((length,), dtype=l.dtype)
    for i in xrange(res.shape[0]):
        c = Counter(l[i*stride:i*stride + stack])
        tmp = sorted(c.items(), key=lambda x: x[1])
        best = tmp[-1][0]
        if omit_na_if_possible and best == 'not_annotated' and len(tmp) > 1:
            best = tmp[-2][0]
        res[i] = best
    return res


def more_than_half_frame_labels(l, length, stride, stack):
    res = np.ndarray((length,), dtype=l.dtype)
    for i in xrange(res.shape[0]):
        c = Counter(l[i*stride:i*stride + stack])
        tmp = sorted(c.items(), key=lambda x: x[1])
        best = "not_annotated"
        if tmp[-1][1] >= stride:
            best = tmp[-1][0]
        res[i] = best
    return res


def more_than_ratio_frame_labels(l, length, stride, stack, ratio):
    res = np.ndarray((length,), dtype=l.dtype)
    for i in xrange(res.shape[0]):
        c = Counter(l[i*stride:i*stride + stack])
        tmp = sorted(c.items(), key=lambda x: x[1])
        best = "not_annotated"
        if tmp[-1][1] > ratio * stack:
            best = tmp[-1][0]
        res[i] = best
    return res


def load_from_npz(fname):
    d = np.load(fname)
    fbanks = d['fbanks']
    labels = d['labels']
    strided = d['strided']
    labels_strided = d['labels_strided']
    stride = d['stride']
    print "stride", stride
    print labels_strided.shape[0]
    print strided.shape
    labels_set = set(labels_strided)
    print labels_set
    print "loaded:", fname
    return fbanks, labels, strided, labels_strided, stride, labels_set


if __name__ == "__main__":

    sounds = []
    strided_sounds = []
    all_annotations = []
    all_strided_annotations = []

    for wavfname in glob.glob(DATASET_PATH):
        annotfname = '.'.join(wavfname.split('.')[:-1]) + '.TextGrid'
        if not len(glob.glob(annotfname)):
            "no textgrid for this file", wavfname, annotfname  # TODO do something
            continue

        srate, sound = wavfile.read(wavfname)
        fbanks = Mel(nfilt=nfbanks,          # nb of filters in mel bank
                alpha=0.97,             # pre-emphasis
                fs=srate,               # sampling rate
                frate=rfbanks,          # frame rate
                wlen=wfbanks,           # window length
                nfft=1024,              # length of dft => 512 is not enough and produces a glitch
                mel_deltas=False,       # speed
                mel_deltasdeltas=False  # acceleration
                )
        fbank = fbanks.transform(sound)[0]  # first dimension is for
                                            # deltas & deltasdeltas
        if not GLOBAL_NORMALIZE:
            fbank = (fbank - fbank.mean(axis=0)) / fbank.std(axis=0)
        sounds.append(fbank)
        print fbank.shape

        annotations = ["not_annotated" for _ in xrange(fbank.shape[0])]
        annotations = np.array(annotations)
        d = parse_textgrid(annotfname)
        for label, timings in d.iteritems():
            for start, end in timings:
                annotations[start*rfbanks:end*rfbanks] = label
        all_annotations.append(annotations)
        print annotations.shape

        strided = np.zeros(((fbank.shape[0] + STRIDE - 1)/STRIDE,
            fbank.shape[1] * STACK))
        for i in xrange(strided.shape[0]):
            tmp = fbank[i*STRIDE:i*STRIDE + STACK].flatten()
            if tmp.shape[0] < STACK * nfbanks:
                tmp = np.pad(tmp, (0, STACK*nfbanks - tmp.shape[0]), mode="constant")
            strided[i] = tmp
        if not GLOBAL_NORMALIZE:
            strided = (strided - strided.mean(axis=0)) / strided.std(axis=0)
        strided_sounds.append(strided)
        print strided.shape

        #labels_strided = vote_labels(annotations, strided.shape[0], STRIDE, STACK)
        labels_strided = more_than_ratio_frame_labels(annotations, 
                strided.shape[0], STRIDE, STACK, ratio=0.5)
        print labels_strided.shape
        all_strided_annotations.append(labels_strided)

    sounds = np.concatenate(sounds, axis=0)
    strided_sounds = np.concatenate(strided_sounds, axis=0)
    if GLOBAL_NORMALIZE:
        sounds = (sounds - sounds.mean(axis=0)) / sounds.std(axis=0)
        strided_sounds = (strided_sounds - strided_sounds.mean(axis=0)) / strided_sounds.std(axis=0)
    print sounds.shape
    print strided_sounds.shape
    np.savez_compressed(OUTPUT_FILE, fbanks=sounds,
            labels=np.concatenate(all_annotations, axis=0),
            strided=strided_sounds,
            labels_strided=np.concatenate(all_strided_annotations),
            stride=STRIDE,
            stack=STACK,
            nfbanks=nfbanks,
            wfbanks=wfbanks,
            rfbanks=rfbanks)


