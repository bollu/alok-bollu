#!/usr/bin/env python3
import subprocess
import sys
import scipy.stats
import numpy as np

# xh = "cross" "h" = "cross entropy"
# symxh = "symmetric" "cross" "h" = "symmetric cross entropy"
if (len(sys.argv) != 4): raise RuntimeError("usage: %s <word2vec-embed-path> <composition-text-file-path> <kl/xh/symxh>", sys.argv[0])

TEMPFILEPATH="dump-composition.temp.txt"
tempfile=open(TEMPFILEPATH, "w");
subprocess.call(["./compute-composition", sys.argv[1], sys.argv[2], sys.argv[3]], stdout=tempfile)
tempfile.close()

with open(TEMPFILEPATH, "r") as f:
    pairs = np.asarray([ [float(w) for w in l.strip().split()] for l in f.readlines()])
    print(pairs[0])
    print("spearman correlation: %s" % (scipy.stats.spearmanr(pairs[:, 0], pairs[:, 1]), ))
    print("pearson correlation: %s" % (scipy.stats.pearsonr(pairs[:, 0], pairs[:, 1]), ))
