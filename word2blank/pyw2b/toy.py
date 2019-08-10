#!/usr/bin/env python3
import argparse
import sys
import torch.multiprocessing as tm
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
import threading
from numpy import random

CORPUSPATH="text1"
NUMTHREADS = 1
BATCHSIZE = 1000
WINDOWSIZE = 4
EPOCHS = 300
EMBEDSIZE = 3
STARTING_ALPHA = 1
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device("cpu")

with open(CORPUSPATH, "r") as f:
    corpus = np.array(f.read().split())[:1000]
print("corpus:\n", corpus)
print("corpus length: ", len(corpus))

print("building vocab...")
vocab = set(corpus)
print(  "vocab length: ", len(vocab))

# match each word in the vocabulary to an index
print("creating vocabIxs")
vocabIxs = dict(zip(vocab, range(len(vocab))))

print("creating neighbours tensor")
# vocabNeighbours = torch.zeros([len(vocab), len(vocab)], dtype=torch.bool)
vocabNeighbours = np.zeros([len(vocab), len(vocab)], dtype=np.bool)

print("created ixedcorpus")
ixedcorpus = np.zeros(len(corpus), dtype=np.int)

print("ixing corpus")
for i in range(len(corpus)):
    ixedcorpus[i] = vocabIxs[corpus[i]]

print("ixed corpus: ", ixedcorpus)
part = max(len(corpus) // 10, 1)
for i in range(len(corpus)):
    iix = ixedcorpus[i]
    if i % part == 0: print("*")
    left = max(i-WINDOWSIZE, 0)
    right = min(i+WINDOWSIZE, len(vocab) - 1)
    vocabNeighbours[iix][np.arange(left, right)] = True

print("vocab neighbours: ", vocabNeighbours)
vocabNeighbours = torch.tensor(vocabNeighbours, device=DEVICE, dtype=torch.bool)
print("sent vocab neighbours to device: ", vocabNeighbours)

print("creating vecs")
vecs = torch.randn([len(vocab) *2, EMBEDSIZE], device=DEVICE, requires_grad=True, dtype=torch.float)

print("corpus length: %s" % len(corpus))
print("vocab %s" % len(vocab))

def normalize(v):
    return v / torch.sqrt(v.dot(v))

# get the embedded representation for these words
def embedtensor(embeddict, ws):
    return torch.stack([embeddict[w] for w in ws])
    

# fis, cis = 1 row, BATCH columns
def trainsample(fis, cis, label, alpha):
    i = 0
    while i < BATCHSIZE:
        i += 1
        f = vecs[wordixs[corpus[i]]]
        c = vecs[len(vocab) + wordixs[corpus[i]]]
        d = f.dot(c)

        loss = (label - d) * (label - d)
        loss.backward()

        # f.data.sub_(f.grad.data * alpha)
        # f.grad.data.zero_()
        # 
        # c.data.sub_(c.grad.data * alpha)
        # c.grad.data.zero_()
    vecs.data.sub_(vecs.grad.data * alpha)
    vecs.grad.data.zero_()
    return 0


def perthread(tid):
    # iterate through
    count = 0
    last_seen_words = 0
    total_loss = 0
    PER_THREAD_SIZE = math.ceil(len(corpus) / NUMTHREADS)

    start = tid * PER_THREAD_SIZE
    end = min((tid + 1) * PER_THREAD_SIZE, len(corpus) - 1)

    cur_corpus = corpus[start:end]
    per_epoch = max(1, len(cur_corpus) // BATCHSIZE)
    print("%d | per epoch: %d" % (tid, per_epoch))
    total = EPOCHS * per_epoch
    alpha = STARTING_ALPHA

    for epoch in range(EPOCHS):
        print("%d | vecs: %s" % (epoch, vecs))
        for b in range(per_epoch):
            # take (1000x1000)
            randixs = np.random.randint(start, end, BATCHSIZE)
            # get the words corresponding to the corpus
            fixs = ixedcorpus[randixs]

            f = vecs[fixs]
            c = vecs[fixs + len(vocab)]
            
            # y = contains all dot products of each fixs with other fixs
            y = torch.sigmoid(torch.matmul(f, torch.transpose(c, 0, 1)))

            print("y(", y.shape, ") | y: ", y)

            if b % max((per_epoch // 10), 1) == 0:
                print("%d | epoch: %d | percent: %d%%" % (tid, epoch, (b / per_epoch) * 100))

            # get the words and fill in the dot products
            labels = vocabNeighbours[fixs].transpose(0, 1)[fixs].transpose(0, 1)

            # loss = \sum_ij (labels_ij - y_ij)^2 
            loss = labels.float() - y
            loss = loss * loss

            # backprop loss
            loss = loss.sum(0).sum(0)
            loss.backward()

            print("loss: ", loss)
            print("vec before grad ", vecs)
            print("grad: ", vecs.grad / (BATCHSIZE * BATCHSIZE))

            # gradient descent
            vecs.data.sub_(vecs.grad.data * alpha / (BATCHSIZE * BATCHSIZE))
            vecs.grad.data.zero_()
            print("vec after grad ", vecs)



        print("%d | done with epoch: %d" % (tid, epoch))

def trainmodel():
    ts = [threading.Thread(target=perthread, args=(i,)) for i in range(NUMTHREADS)]
    for t in ts:
        t.start()

    for t in ts:
        t.join()

    with open("model.out", "wb") as f:
        pickle.dump(vecs, f)

# assumes syn0 is initialized
def test():
    session = PromptSession()
    with torch.no_grad():
        while True:
            word = session.prompt(">")
            if word not in vocab:
                print("%s out of corpus." % word)
                continue
            dots = []
            v = normalize(vecs[vocabIxs[word]])
            print("v: %s" % v)
            for word2 in vocab:
                v2 = normalize(vecs[vocabIxs[word2]]) 
                dots.append((word2, v.dot(v2)))
            dots.sort(key=lambda wordvec: wordvec[1])
            for (word, d) in dots[::-1][:10]:
                print("%s %20.4f" % (word.rjust(20, " "), d))

def loadmodel():
    with open("model.out", "rb") as f:
        global syn0
        syn0 = pickle.load(f)

if __name__ == "__main__":
    trainmodel()
    test()
