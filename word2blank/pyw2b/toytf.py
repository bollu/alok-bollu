#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import argparse
import sys
import datetime
import os
import os.path
import itertools
import math
import pudb
import pickle
from prompt_toolkit import PromptSession

CORPUSPATH="text1"
NTHREADS = 40
BATCHSIZE = 100
WINDOWSIZE = 4
NNEGSAMPLES = 15
EPOCHS = 15
EMBEDSIZE = 2
STARTING_ALPHA = 0.01
DEVICE = tf.device("cpu")

with open(CORPUSPATH, "r") as f:
    corpus = np.array(f.read().split())



vocab = set(corpus[:])
init = tf.glorot_normal_initializer
print("initializing syn0, syn1neg")
SHAPE = [len(vocab)*2, EMBEDSIZE]
vectors = tf.Variable(shape=SHAPE, initial_value=init(SHAPE), dtype=tf.float32)

print("corpus length: %s" % len(corpus))
print("vocab %s" % len(vocab))


def trainmodel():
    count = 0
    last_seen_words = 0
    total_loss = 0
    for _ in range(EPOCHS):
        for fis in np.array_split(np.arange(0,len(corpus)-1), size // BATCHSIZE):
            # positive sample training
            # context words are focus words, with random perturbation
            cis = fis + np.random.randint(-WINDOWSIZE,+WINDOWSIZE, size=fis.size, dtype=np.int32)
            # make sure this is within bounds.
            cis = np.clip(cis, 0, len(corpus) - 1)
            total_loss += trainsample(fis, cis, label1, alpha)

            # # random vector
            # cis = np.random.randint(len(corpus), size=fis.size)
            # total_loss += trainsample(fis, cis, label0, alpha)

            count += 1
            last_seen_words += 1

            if (last_seen_words >= 1e2):
                sys.stdout.write("\r%d: %f%% Loss: %4.2f\n" % (ix, (float(count) / total) * 100, total_loss))
                sys.stdout.flush()
                last_seen_words = 0
                total_loss = 0

    with open("model.out", "wb") as f:
        pickle.dump(syn0, f)

if __name__ == "__main__":
    trainmodel()
