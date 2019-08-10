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
from prompt_toolkit import PromptSession

CORPUSPATH="text1"
BATCHSIZE = 100
WINDOWSIZE = 4
NNEGSAMPLES = 15
EPOCHS = 15
EMBEDSIZE = 2
STARTING_ALPHA = 0.01
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')

with open(CORPUSPATH, "r") as f:
    corpus = np.array(f.read().split())

vocab = set(corpus)

print("creating syn0...")
syn0 = { v : torch.randn(EMBEDSIZE, device=DEVICE, requires_grad=True, dtype=torch.float)
                for v in vocab }

print("creating syn1neg...")
syn1neg = { v : torch.randn(EMBEDSIZE, device=DEVICE, requires_grad=True, dtype=torch.float)
                for v in vocab }

print("corpus length: %s" % len(corpus))
print("vocab %s" % len(vocab))

def normalize(v):
    return v / torch.sqrt(v.dot(v))

# get the embedded representation for these words
def embedtensor(embeddict, ws):
    return torch.stack([embeddict[w] for w in ws])
    

# fis, cis = 1 row, BATCH columns
def trainsample(fis, cis, label, alpha):
    # vec_fis, vec_cis = BATCH rows, EMBED coluns
    vec_fis = [syn0[corpus[i]] for i in fis]
    vec_cis = [syn1neg[corpus[i]] for i in cis]

    dots = torch.stack([f.dot(c) for (f,c) in zip(vec_fis, vec_cis)])
    loss =  (label - dots)
    # loss^2
    loss = loss.dot(loss)
    loss.backward()

    # need to index using data
    for v in vec_fis + vec_cis:
        v.data.sub_(v.grad.data * alpha)
        # zero out the loss
        v.grad.data.zero_()

    return float(loss)


def trainmodel():
    # iterate through
    count = 0
    last_seen_words = 0
    total_loss = 0
    total = EPOCHS * len(corpus) // BATCHSIZE
    alpha = STARTING_ALPHA
    for _ in range(EPOCHS):
        # fi = focus index
        for fis in np.array_split(np.arange(len(corpus), dtype=np.int32), len(corpus) // BATCHSIZE):
            # positive sample training
            # context words are focus words, with random perturbation
            cis = fis + np.random.randint(-WINDOWSIZE,+WINDOWSIZE, size=fis.size,
                                          dtype=np.int32)
            # make sure this is within bounds.
            cis = np.clip(cis, 0, len(corpus) - 1)
            total_loss += trainsample(fis, cis, torch.tensor([1.0], device=DEVICE), alpha)

            # random vector
            cis = np.random.randint(len(corpus), size=fis.size)
            total_loss += trainsample(fis, cis, torch.tensor([0.0], device=DEVICE), alpha)

            count += 1
            last_seen_words += 1

            if (last_seen_words > 1e3):
                sys.stdout.write("\r%f%% Loss: %4.2f\n" % ((float(count) / total) * 100, total_loss))
                sys.stdout.flush()
                last_seen_words = 0
                total_loss = 0

    with open("model.out", "wb") as f:
        pickle.dump(syn0, f)

# assumes syn0 is initialized
def test():
    session = PromptSession()
    with torch.no_grad():
        while True:
            word = session.prompt(">")
            if word not in syn0:
                print("%s out of corpus." % word)
                continue
            dots = []
            v = normalize(syn0[word])
            print("v: %s" % v)
            for (word2, v2) in syn0.items():
                v2 = normalize(v2) 
                dots.append((word2, v.dot(v2)))
            dots.sort(key=lambda wordvec: wordvec[1])
            for (word, vec) in dots[:10]:
                print("%s %20.4f" % (word.rjust(20, " "), vec))

def loadmodel():
    with open("model.out", "rb") as f:
        global syn0
        syn0 = pickle.load(f)

if __name__ == "__main__":
    trainmodel()
    test()
