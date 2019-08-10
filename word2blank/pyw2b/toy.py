#!/usr/bin/env python3
import argparse
import sys
import datetime
import os
import os.path
import itertools
import torch
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import math
import pudb
import tabulate
from random import sample
import pickle

CORPUSPATH="text1"
BATCHSIZE = 100
WINDOWSIZE = 4
NNEGSAMPLES = 15
EPOCHS = 15
EMBEDSIZE = 3
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')

with open(CORPUSPATH, "r") as f:
    corpus = np.array(f.read().split())

vocab = set(corpus)

print("creating syn0...")
syn0 = { v : nn.Parameter(Variable(torch.randn(EMBEDSIZE, device=DEVICE)))
                for v in vocab }

print("creating syn1neg...")
syn1neg = { v : nn.Parameter(Variable(torch.zeros(EMBEDSIZE, device=DEVICE)))
                for v in vocab }

print("corpus length: %s" % len(corpus))
print("vocab %s" % len(vocab))

def embedtensor(embeddict, ws):
    return torch.stack([embeddict[w] for w in ws])
    

# fis, cis = 1 row, BATCH columns
def train(fis, cis, label):
    # vec_fis, vec_cis = BATCH rows, EMBED coluns
    vec_fis = embedtensor(syn0, corpus[fis])
    vec_cis = embedtensor(syn1neg, corpus[cis])
    out = torch.einsum('...i,...i->...', vec_fis, vec_cis)
    return torch.sigmoid(label - out)

# iterate through
count = 0
last_seen_words = 0

total = EPOCHS * len(corpus) // BATCHSIZE
for _ in range(EPOCHS):
    # fi = focus index
    for fis in np.array_split(np.arange(len(corpus), dtype=np.int32), len(corpus) // BATCHSIZE):
        loss = 0
        # positive sample training
        # context words are focus words, with random perturbation
        cis = fis + np.random.randint(-WINDOWSIZE,+WINDOWSIZE, size=fis.size,
                                      dtype=np.int32)
        # make sure this is within bounds.
        cis = np.clip(cis, 0, len(corpus) - 1)
        loss += train(fis, cis, torch.tensor([1.0], device=DEVICE))

        # random vector
        cis = np.random.randint(len(corpus), size=fis.size)
        loss += train(fis, cis, torch.tensor([0.0], device=DEVICE))

        count += 1
        last_seen_words += 1

        if (last_seen_words > 1e3):
            sys.stdout.write("\r%f%%\n" % ((float(count) / total) * 100))
            sys.stdout.flush()
            last_seen_words = 0

with open("model.out", "wb") as f:
    pickle.dump(syn0, f)
