from spectral import Mel  # you need https://github.com/mwv/spectral
import numpy as np
from scipy.io import wavfile
from collections import defaultdict
import glob

def parse_textgrid(fname):
    with open(fname) as f:
        l = f.readline()
    res = defaultdict(lambda: [])
    for i, line in enumerate(l):
        if 'text = ' in line:
            label = line.split('=')[1].strip(' "')
            if label != "":
                res[label].append((float(l[i-2].split('=')[1]),
                    float(l[i-1].split('=')[1])))
    return res


nfbanks = 40
wfbanks = 0.025 # 25ms
rfbanks = 100 # 10ms

all_sounds = []
all_annotations = []

for wavfname in glob.iglob("../Blue Monkey/*.WAV"):
    if "0B." in wavfname: # bugged file TODO
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
    all_sounds.append(fbank)
    print fbank.shape

    annotations = ["not_annotated" for _ in xrange(fbank.shape[0])]
    annotations = np.array(annotations)
    try:
        annotfname = '.'.join(wavfname.split('.')[:-1]) + '.TextGrid'
        d = parse_textgrid(annotfname)
        for label, timings in d.iteritems():
            for start, end in timings:
                annotations[start*rfbanks:end*rfbanks] = label
    except IOError:
        "no textgrid for this file", wavfname, annotfname  # TODO do something
    all_annotations.append(annotations)


np.savez("blue_monkeys.npz", fbanks=np.concatenate(all_sounds, axis=0),
        labels=np.concatenate(all_annotations, axis=0))




