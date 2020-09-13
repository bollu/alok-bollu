import subprocess
import sys
import scipy.stats
import numpy as np
import os

if (len(sys.argv) != 3): 
    raise RuntimeError("usage: \n%s <word2vec-embed-path> <simlex-text-file-path> "%(sys.argv[0]))

TEMPFILEPATH="output.txt"
# if os.path.exists(TEMPFILEPATH): os.remove(TEMPFILEPATH)

subprocess.call(["./simlex-accuracy", 
    sys.argv[1], sys.argv[2], TEMPFILEPATH])

with open(TEMPFILEPATH, "r") as f:
    pairs = np.asarray([ [float(w) for w in l.strip().split()] for l in f.readlines()])
    print(pairs[-2:])
    print("spearman correlation: %s" % (scipy.stats.spearmanr(pairs[:, 0], pairs[:, 1]), ))