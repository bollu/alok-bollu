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
import threading

CORPUSPATH="text1"
NTHREADS = 1
BATCHSIZE = 1000
WINDOWSIZE = 4
NNEGSAMPLES = 15
EPOCHS = 15
EMBEDSIZE = 100
STARTING_ALPHA = 0.01
# DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device("cpu")

with open(CORPUSPATH, "r") as f:
    corpus = np.array(f.read().split())
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
part = len(corpus) // 10
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

def trainthread(ix):
    size = len(corpus) // NTHREADS
    begin = min(size * ix, len(corpus) - 1)
    end = min(size *(ix+1) - 1, len(corpus) - 1)

    total = (EPOCHS * size) // BATCHSIZE
    count = 0

    last_seen_words = 0
    total_loss = 0


    alpha = STARTING_ALPHA

    label1 = torch.ones(1, device=DEVICE)
    label0 = torch.zeros(1, device=DEVICE)

        # for fis in np.array_split(np.arange(begin,end), size // BATCHSIZE):
        #     # positive sample training
        #     # context words are focus words, with random perturbation
        #     cis = fis + np.random.randint(-WINDOWSIZE,+WINDOWSIZE, size=fis.size, dtype=np.int32)
        #     # make sure this is within bounds.
        #     cis = np.clip(cis, 0, len(corpus) - 1)
        #     total_loss += trainsample(fis, cis, label1, alpha)

        #     # # random vector
        #     # cis = np.random.randint(len(corpus), size=fis.size)
        #     # total_loss += trainsample(fis, cis, label0, alpha)

        #     count += 1
        #     last_seen_words += 1

        #     if (last_seen_words >= 1e1):
        #         sys.stdout.write("\r%d: %f%% Loss: %4.2f\n" % (ix, (float(count) / total) * 100, total_loss))
        #         sys.stdout.flush()
        #         last_seen_words = 0
        #         total_loss = 0


def trainmodel():
    # iterate through
    count = 0
    last_seen_words = 0
    total_loss = 0
    total = EPOCHS * len(corpus) // BATCHSIZE
    per_epoch = len(corpus) // BATCHSIZE
    alpha = STARTING_ALPHA
    print("per_epoch: ", per_epoch)

    for epoch  in range(EPOCHS):
        for b in range(per_epoch):
            # take (1000x1000)
            randixs = list(range(BATCHSIZE))
            # get the words corresponding to the corpus
            fixs = ixedcorpus[randixs]

            f = vecs[fixs]
            c = vecs[fixs + len(vocab)]
            
            y = torch.matmul(f, torch.transpose(c, 0, 1))

            # get the words and fill in the dot products
            print("filling in dot products in label...")
            labels = vocabNeighbours[fixs].transpose(0, 1)[fixs].transpose(0, 1)
            print("  filled.")

            # loss = \sum_ij (labels_ij - y_ij)^2 
            loss = labels.float() - y
            loss = loss * loss
            print("summing over pointwise loss...")
            loss = loss.sum(0).sum(0)
            print("loss:", loss)

            print("backpropping loss...")
            loss.backward()

            print("updating vec by gradient")
            vecs.data.sub_(vecs.grad.data * alpha)
            vecs.grad.data.zero_()


        print("    done with epoch: %d" % epoch)

    with open("model.out", "wb") as f:
        pickle.dump(vecs, f)

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
