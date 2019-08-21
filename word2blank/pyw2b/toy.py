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

EPOCHS = 100
EMBEDSIZE = 3
NVECS = 10
NNEGSAMPLES = 15
STARTING_ALPHA = 0.001
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')

posvecs = torch.randn([NVECS, EMBEDSIZE], device=DEVICE, requires_grad=True, dtype=torch.float)
negvecs = torch.randn([NVECS, EMBEDSIZE], device=DEVICE, requires_grad=True, dtype=torch.float)
targetDots = torch.randn([NVECS, NVECS], device=DEVICE, requires_grad=False, dtype=torch.float)


def normalize(v):
    return v / torch.sqrt(v.dot(v))


def calculate_loss():
    l = 0
    with torch.no_grad():
        for i in range (NVECS):
            for j in range (NVECS):
                score = (targetDots[i][j] - posvecs[i].dot(posvecs[j]))
                l += score * score
    return l

def trainW2V():
    total_loss = 0
    last_nvecs = 0
    nvecs = 0

    total = EPOCHS * (NVECS * NVECS)
    alpha = STARTING_ALPHA

    for epoch in range(EPOCHS):
        for i in range(NVECS):
            for j in range(NVECS):
                nvecs += 1
                if i == j: continue

                if nvecs - last_nvecs > 100:
                    last_nvecs = nvecs
                    percent_done = (nvecs / total) * 100.0
                    print("%4.2f %% | total loss: %4.2f" % (percent_done, calculate_loss()))
                    total_loss = 0

                d = posvecs[i].dot(negvecs[j])
                score = (targetDots[i][j] - d)
                loss = score * score
                total_loss += loss.data

                grad = loss.backward()
                posvecs.data.sub_(posvecs.grad.data * alpha)
                posvecs.grad.data.zero_()

                negvecs.data.sub_(negvecs.grad.data * alpha)
                negvecs.grad.data.zero_()



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
    trainW2V()
