#!/usr/bin/env python3
from collections import Counter
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import numpy as np
import signal
import sys
import click
import datetime 
import math
TEXT = """we are about to study the idea of a computational process.
computational processes are abstract beings that inhabit computers.
as they evolve, processes manipulate other abstract things called data.
the evolution of a process is directed by a pattern of rules                                                                      called a program.
people create programs to direct processes.
in effect, we conjure the spirits of the computer with our spells.""".split()                                                                
STOPWORDS = set(["the", "as", "in", "is", "to", "of", "that", "they", "a",
                 "with", "by"])
TEXT = list(filter(lambda w: w not in STOPWORDS, TEXT))

EPOCHS = 30
BATCHSIZE = 2
EMBEDSIZE = 5
VOCAB = set(TEXT)
WINDOWSIZE = 2

VOCABSIZE = len(VOCAB)

i2w = dict(enumerate(VOCAB))
w2i = { v: k for (k, v) in i2w.items() }

def hot(ws):
    """
    hot vector for each word in ws
    """
    v = Variable(torch.zeros(VOCABSIZE).float())
    for w in ws: v[w2i[w]] = 1.0
    return v

DATA = []
for i in range((len(TEXT) - 2 * WINDOWSIZE)):
    ix = i + WINDOWSIZE

    wsfocus = [TEXT[ix]]
    wsctx = [TEXT[ix + deltaix] for deltaix in range(-WINDOWSIZE, WINDOWSIZE + 1)]
    DATA.append(torch.stack([hot(wsfocus), hot(wsctx)]))
DATA = torch.stack(DATA)

def batch(xs):
    ix = 0
    while ix + BATCHSIZE < len(xs):
        data = xs[ix:ix+BATCHSIZE]
        ix += BATCHSIZE
        yield data

# model matrix
INM = nn.Parameter(torch.randn(VOCABSIZE, EMBEDSIZE))


def norm(v, w, metric):
    dot = torch.mm(torch.mm(v.view(1, -1), metric), w.view(-1, 1))
    return dot

def normalize(v, metric):
    normsq = norm(v, v, metric).item()
    return v.float() / math.sqrt(normsq)

def cosinesim(v, w, metric):
    v = normalize(v, metric)
    w = normalize(w, metric)
    return norm(v, w, metric)

def dots(vs, metric):
    # M = [VOCABSIZE x EMBEDSIZE]
    # vs = [BATCHSIZE x EMBEDSIZE]
    # metric = [EMBEDSIZE x EMBEDSIZE]

    # outs = [BATCHSIZE x VOCABSIZE]
    outs = torch.zeros([BATCHSIZE, VOCABSIZE])
    for vix in range(BATCHSIZE):
        # v = [1 x EMBEDSIZE]
        v = vs[vix, :]
        for wix in range(VOCABSIZE):
            # w = [EMBEDSIZE x 1]
            w = INM[wix, :]
            # [1 x EMBEDSIZE] x [EMBEDSIZE x EMBEDSIZE] x [EMBEDSIZE x 1] = [1x1]
            outs[vix][wix] = cosinesim(v, w, metric)
    return outs

optimizer = optim.SGD([INM], lr=0.01)
loss = nn.MSELoss()

# metric, currently identity matrix
metric = torch.eye(EMBEDSIZE)

for _ in range(EPOCHS):
    for train in batch(DATA):
        # [BATCHSIZE x VOCABSIZE]
        xs = train[:, 0]
        # [BATCHSIZE x VOCABSIZE]
        ysopt = train[:, 1]

        optimizer.zero_grad()   # zero the gradient buffers
        # embedded vectors of the batch vectors
        # [BATCHSIZE x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [BATCHSIZE x EMBEDSIZE]
        xsembeds = torch.mm(xs, INM)

        # [BATCHSIZE x VOCABSIZE]
        ysembeds = dots(xsembeds, metric)

        l = loss(ysembeds, ysopt)
        l.backward()
        optimizer.step()

# testing
for w in VOCAB:
    # [1 x VOCABSIZE] 
    whot = hot([w])
    # [1 x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [1 x EMBEDSIZE]
    wembed = torch.mm(whot.view(1, -1), INM)

    dots = torch.zeros(VOCABSIZE).float()
    for ix in range(VOCABSIZE):
        v = INM[ix, :]
        # print("*v: %s\n*wembed: %s\n*metric: %s"% (v, wembed, metric))
        dots[ix] = cosinesim(v, wembed, metric)

    wordweights = [(i2w[i], dots[i].item()) for i in range(VOCABSIZE)]
    wordweights.sort(key=lambda (w, dot): dot, reverse=True)
    wordweights = wordweights[:10]

    print("* %s"% w)
    for (word, weight) in wordweights:
        print("\t%s: %s" % (word, weight))
