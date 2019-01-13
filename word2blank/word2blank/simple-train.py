#!/usr/bin/env python
from __future__ import print_function
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
import progressbar

class TimeLogger:
    def __init__(self):
        pass
    def start(self, toprint):
        self.t = datetime.datetime.now()
        print(str(toprint) + "...", end="")

    def end(self, toprint=None):
        now = datetime.datetime.now()
        if(toprint): print(toprint)
        print("Done. time: %s" % (now - self.t))
        if (toprint): print("--")

LOGGER = TimeLogger()

# setup device
LOGGER.start("setting up device")
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')
LOGGER.end("device: %s" % DEVICE)

def load_corpus():
    return  """we are about to study the idea of a computational process.
     computational processes are abstract beings that inhabit computers.
     as they evolve, processes manipulate other abstract things called data.
     the evolution of a process is directed by a pattern of rules called a program.
     people create programs to direct processes.
     in effect, we conjure the spirits of the computer with our spells.""".split()                                                                
    CORPUS_NAME = "text8"
    LOGGER.start("loading corpus: %s" % CORPUS_NAME)
    try:
        sys.path.insert(0, api.base_dir)
        module = __import__(CORPUS_NAME)
        CORPUS = module.load_data()
    except Exception as e:
        print("unable to find text8 locally.\nERROR: %s" % (e, ))
        print("Downloading using gensim-data...")
        CORPUS = api.load(CORPUS_NAME)
        print("Done.")

    text = list(CORPUS)[0]
    LOGGER.end()
    return text

TEXT = load_corpus()

STOPWORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
             "you", "your", "yours", "yourself", "yourselves", "he", "him", 
             "his", "himself", "she", "her", "hers", "herself", "it", "its", 
             "itself", "they", "them", "their", "theirs", "themselves", 
             "what", "which", "who", "whom", "this", "that", "these", 
             "those", "am", "is", "are", "was", "were", "be", "been", 
             "being", "have", "has", "had", "having", "do", "does", 
             "did", "doing", "a", "an", "the", "and", "but", 
             "if", "or", "because", "as", "until", "while", 
             "of", "at", "by", "for", "with", "about", 
             "against", "between", "into", "through", "during", "before", 
             "after", "above", "below", "to", "from", "up", "down", "in", 
             "out", "on", "off", "over", "under", "again", "further", "then", 
             "once", "here", "there", "when", "where", "why", "how", "all", 
             "any", "both", "each", "few", "more", "most", "other", "some", 
             "such", "no", "nor", "not", "only", "own", "same", "so", 
             "than", "too", "very", "s", "t", "can", "will", "just", "don", 
             "should", "now"])
LOGGER.start("filtering stopwords")
TEXT = list(filter(lambda w: w not in STOPWORDS, TEXT))
LOGGER.end()

EPOCHS = 2
BATCHSIZE = 2
EMBEDSIZE = 100
LEARNING_RATE = 0.025
VOCAB = set(TEXT)
WINDOWSIZE = 2

VOCABSIZE = len(VOCAB)

LOGGER.start("creating i2w, w2i")
i2w = dict(enumerate(VOCAB))
w2i = { v: k for (k, v) in i2w.items() }
LOGGER.end()

def hot(ws):
    """
    hot vector for each word in ws
    """
    v = Variable(torch.zeros(VOCABSIZE).float())
    for w in ws: v[w2i[w]] = 1.0
    return v

LOGGER.start("creating DATA")
DATA = []
for i in range((len(TEXT) - 2 * WINDOWSIZE)):
    ix = i + WINDOWSIZE

    wsfocus = [TEXT[ix]]
    wsctx = [TEXT[ix + deltaix] for deltaix in range(-WINDOWSIZE, WINDOWSIZE + 1)]
    DATA.append(torch.stack([hot(wsfocus), hot(wsctx)]))
DATA = torch.stack(DATA)
LOGGER.end()

def batch(xs):
    ix = 0
    while ix + BATCHSIZE < len(xs):
        data = xs[ix:ix+BATCHSIZE]
        ix += BATCHSIZE
        yield data

# model matrix
# EMBEDM = nn.Parameter(torch.randn(VOCABSIZE, EMBEDSIZE)).to(DEVICE)
LOGGER.start("creating EMBEDM")
EMBEDM = nn.Parameter(Variable(torch.randn(VOCABSIZE, EMBEDSIZE).to(DEVICE), requires_grad=True))
LOGGER.end()

def norm(v, w, metric):
    dot = torch.mm(torch.mm(v.view(1, -1), metric), w.view(-1, 1))
    return dot


def dots(vs, ws, metric):
    """Take the dot product of each element in vs with elements in ws"""
    # vs = [S1 x EMBEDSIZE]
    # ws = [S2 x EMBEDSIZE] | ws^t = [EMBEDSIZE x S2]
    # metric = [EMBEDSIZE x EMBEDSIZE]
    # vs * metric = [S1 x EMBEDSIZE]
    # vs * metric * ws^t = [S1 x EMBEDSIZE] x [EMBEDSIZE x S1] = [S1 x S2]

    return torch.mm(torch.mm(vs, metric), ws.t())


    # outs = [BATCHSIZE x VOCABSIZE]
    # outs = torch.zeros([BATCHSIZE, VOCABSIZE])
    # for vix in range(BATCHSIZE):
    #     # v = [1 x EMBEDSIZE]
    #     v = vs[vix, :]
    #     for wix in range(VOCABSIZE):
    #         # w = [EMBEDSIZE x 1]
    #         w = EMBEDM[wix, :]
    #         # [1 x EMBEDSIZE] x [EMBEDSIZE x EMBEDSIZE] x [EMBEDSIZE x 1] = [1x1]
    #         outs[vix][wix] = cosinesim(v, w, metric)
    # return outs
    
def normalize(vs, metric):
    # vs = [S1 x EMBEDSIZE]
    # out = [S1 x EMBEDSIZE]

    # vs_dots_vs = [S1 x S1]
    vs_dots_vs = dots(vs, vs, metric)
    # norm = S1 x 1
    norm = torch.sqrt(torch.diag(vs_dots_vs)).view(-1, 1)
    # normvs = S1 x 1
    normvs =  vs / norm

    ERROR_THRESHOLD = 0.1
    assert(torch.norm(torch.diag(dots(normvs, normvs, metric)) - torch.ones(len(vs)).to(DEVICE)).item() < ERROR_THRESHOLD)
    return normvs

def cosinesim(v, w, metric):
    # vs = [1 x EMBEDSIZE]
    # ws = [1 x EMBEDSIZE]
    # out = [1 x 1]

    # v . w / |v||w| = (v.w)^2 / |v|^2 |w|^2
    # [1 x 1]
    vs_dot_ws = dots(v, w, metric)


    # [1 x 1]
    vs_dot_vs = torch.sqrt(dots(v, v, metric))
    # [1 x 1]
    ws_dot_ws = torch.sqrt(dots(w, w, metric))

    return vs_dot_ws / (vs_dot_vs * ws_dot_ws)


LOGGER.start("creating optimizer and loss function")
optimizer = optim.SGD([EMBEDM], lr=LEARNING_RATE)
loss = nn.MSELoss()
LOGGER.end()

# metric, currently identity matrix
LOGGER.start("creating metric")
METRIC = torch.eye(EMBEDSIZE).to(DEVICE)
LOGGER.end()

# Notice that we do not normalize the vectors in the hidden layer
# when we train them! this is intentional: In general, these vectors don't
# seem to be normalized by most people, so it's weird if we begin to
# normalize them. 
# Read also: what is the meaning of the length of a vector in word2vec?
# https://stackoverflow.com/questions/36034454/what-meaning-does-the-length-of-a-word2vec-vector-have

with progressbar.ProgressBar(max_value=EPOCHS * len(DATA) / BATCHSIZE) as bar:
    for epoch in range(EPOCHS):
        for train in batch(DATA):
            # [BATCHSIZE x VOCABSIZE]
            xs = train[:, 0].to(DEVICE)
            # [BATCHSIZE x VOCABSIZE]
            ysopt = train[:, 1].to(DEVICE)

            optimizer.zero_grad()   # zero the gradient buffers
            # embedded vectors of the batch vectors
            # [BATCHSIZE x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [BATCHSIZE x EMBEDSIZE]
            xsembeds = torch.mm(xs, EMBEDM)

            # dots(BATCHSIZE x EMBEDSIZE], 
            #     [VOCABSIZE x EMBEDSIZE],
            #     [EMBEDSIZE x EMBEDSIZE]) = [BATCHSIZE x VOCABSIZE]
            xs_dots_embeds = dots(xsembeds, EMBEDM, METRIC)

            l = loss(ysopt, xs_dots_embeds)
            l.backward()
            optimizer.step()
            bar.update(bar.value + 1)


def test():
    # [VOCABSIZE x EMBEDSIZE]
    EMBEDNORM = normalize(EMBEDM, METRIC)

    # testing
    for w in VOCAB:
        # [1 x VOCABSIZE] 
        whot = hot([w]).to(DEVICE)
        # [1 x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [1 x EMBEDSIZE]
        wembed = normalize(torch.mm(whot.view(1, -1), EMBEDM), METRIC)

        # dot [1 x EMBEDSIZE] [VOCABSIZE x EMBEDSIZE] = [1 x VOCABSIZE]
        wix2sim = dots(wembed, EMBEDNORM, METRIC)

        wordweights = [(i2w[i], wix2sim[0][i].item()) for i in range(VOCABSIZE)]
        wordweights.sort(key=lambda wdot: wdot[1], reverse=True)

        print("* %s" % w)
        for (word, weight) in wordweights[:4]:
            print("\t%s: %s" % (word, weight))
test()
