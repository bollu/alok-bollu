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
def load_corpus():
    CORPUS_NAME = "text8"
    print ("loading corpus: %s" % (CORPUS_NAME, ))
    try:
        print("loading gensim locally...")
        sys.path.insert(0, api.base_dir)
        module = __import__(CORPUS_NAME)
        CORPUS = module.load_data()
        print("Done.")
    except Exception as e:
        print("unable to find text8 locally.\nERROR: %s" % (e, ))
        print("Downloading using gensim-data...")
        CORPUS = api.load(CORPUS_NAME)
        print("Done.")

    TEXT = list(CORPUS)[0]

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
    return v.float() / torch.sqrt(normsq)

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
    #         w = INM[wix, :]
    #         # [1 x EMBEDSIZE] x [EMBEDSIZE x EMBEDSIZE] x [EMBEDSIZE x 1] = [1x1]
    #         outs[vix][wix] = cosinesim(v, w, metric)
    # return outs

def cosinesim(v, w, metric):
    # vs = [1 x EMBEDSIZE]
    # ws = [1 x EMBEDSIZE]

    # v . w / |v||w| = (v.w)^2 / |v|^2 |w|^2
    # [1 x 1]
    vs_dot_ws = dots(v, w, metric)


    # [1 x 1]
    vs_dot_vs = torch.sqrt(dots(v, v, metric))
    # [1 x 1]
    ws_dot_ws = torch.sqrt(dots(w, w, metric))

    return vs_dot_ws / (vs_dot_vs * ws_dot_ws)


optimizer = optim.SGD([INM], lr=0.01)
loss = nn.MSELoss()

# metric, currently identity matrix
METRIC = torch.eye(EMBEDSIZE)

# Notice that we do not normalize the vectors in the hidden layer
# when we train them! this is intentional: In general, these vectors don't
# seem to be normalized by most people, so it's weird if we begin to
# normalize them. 
# Read also: what is the meaning of the length of a vector in word2vec?
# https://stackoverflow.com/questions/36034454/what-meaning-does-the-length-of-a-word2vec-vector-have

for _ in range(EPOCHS):
    for train in batch(DATA):
        # [BATCHSIZE x VOCABSIZE]
        xs = train[:, 0]
        # [BATCHSIZE x VOCABSIZE]
        ysopt = train[:, 1]
        # [BATCHSIZE x EMBEDSIZE]
        ysopt_embed = torch.mm(ysopt, INM)

        optimizer.zero_grad()   # zero the gradient buffers
        # embedded vectors of the batch vectors
        # [BATCHSIZE x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [BATCHSIZE x EMBEDSIZE]
        xsembeds = torch.mm(xs, INM)

        # dots(BATCHSIZE x EMBEDSIZE], 
        #     [VOCABSIZE x EMBEDSIZE],
        #     [EMBEDSIZE x EMBEDSIZE]) = [BATCHSIZE x EMBEDSIZE]
        xs_dots_embeds = dots(xsembeds, INM, METRIC)

        # [BATCHSIZE x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [BATCHSIZE x EMBEDSIZE]
        ysembed = torch.mm(ysopt, INM)

        l = loss(ysopt_embed, ysembed)
        l.backward()
        optimizer.step()

# testing
for w in VOCAB:
    # [1 x VOCABSIZE] 
    whot = hot([w])
    # [1 x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [1 x EMBEDSIZE]
    wembed = torch.mm(whot.view(1, -1), INM)

    # [1 x VOCABSIZE]
    w2sim = torch.zeros(VOCABSIZE).float()
    # w2sim = cosinesim(wembed, INM, METRIC)
    for ix in range(VOCABSIZE):
        curembed = INM[ix, :].view(1, -1)
        w2sim[ix] = cosinesim(wembed, curembed, METRIC)

    wordweights = [(i2w[i], w2sim[i].item()) for i in range(VOCABSIZE)]
    wordweights.sort(key=lambda (w, dot): dot, reverse=True)
    wordweights = wordweights[:10]

    print("* %s"% w)
    for (word, weight) in wordweights:
        print("\t%s: %s" % (word, weight))
