#!/usr/bin/env python3
import subprocess
import sys
import scipy.stats
import numpy as np

if (len(sys.argv) != 3): raise RuntimeError("usage: %s <word2vec-embed-path> <simlex-text-file-path", sys.argv[0])

TEMPFILEPATH="dump-accuracy-simlex.temp.txt"
subprocess.call(["./dump-accuracy-simlex", sys.argv[1], sys.argv[2], TEMPFILEPATH])

with open(TEMPFILEPATH, "r") as f:
    pairs = np.asarray([ [float(w) for w in l.strip().split()] for l in f.readlines()])
    print(pairs[-10:])
    print(pairs[0])
    print("spearman correlation: %s" % (scipy.stats.spearmanr(pairs[:, 0], pairs[:, 1]), ))
