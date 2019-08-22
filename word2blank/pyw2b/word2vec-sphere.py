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

# Note that this currently uses the retarded euclidian gradient. Change
# to gradient on sphere:
# https://wiseodd.github.io/techblog/2019/02/22/optimization-riemannian-manifolds/


torch.manual_seed(0)

DEBUG = False
EMBEDSIZE = 3
NVECS = 10
NNEGSAMPLES = 15
STARTING_ALPHA = 0.1
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')

posvecs = torch.randn([NVECS, EMBEDSIZE-1], device=DEVICE, requires_grad=True, dtype=torch.float)
negvecs = torch.randn([NVECS, EMBEDSIZE-1], device=DEVICE, requires_grad=True, dtype=torch.float)
targetDots = torch.randn([NVECS, NVECS], device=DEVICE, requires_grad=False, dtype=torch.float)

torch.autograd.set_detect_anomaly(DEBUG)


def normalize(v):
    return v / torch.sqrt(v.dot(v))


def calculate_loss():
    l = 0
    with torch.no_grad():
        for i in range (NVECS):
            for j in range (NVECS):
                score = (targetDots[i][j] - angle2vec(posvecs[i]).dot(angle2vec(posvecs[j])))
                l += score * score
    return l

def angle2vec(angles):
    n = len(angles) + 1

    vec = torch.empty(n, device=DEVICE,requires_grad=False)
    for i in range(n-1):
        prod = 1
        for j in range(0, i):
            prod *= torch.sin(angles[j])
        prod *= torch.cos(angles[i])
        vec[i] = prod

    prod = 1
    for i in range(0, n-1):
        prod *= torch.sin(angles[i])
    vec[n-1] = prod

    # debug code
    if DEBUG:
        with torch.no_grad():
            lensq = 0.0
            for a in vec:
                lensq += a * a
            l = torch.sqrt(lensq)
            assert abs(1 - l) < 1e-2

    return vec

def trainW2V():
    prev_total_loss = 0
    loss_acceleration = 1
    last_nvecs = 0
    nvecs = 0

    t = 0

    while abs(loss_acceleration) > 1e-2:
        for i in range(NVECS):
            for j in range(NVECS):
                t += 0.01
                alpha = STARTING_ALPHA

                nvecs += 1
                if i == j: continue

                if nvecs - last_nvecs > 100:
                    last_nvecs = nvecs
                    cur_total_loss = calculate_loss()
                    loss_acceleration = prev_total_loss - cur_total_loss
                    prev_total_loss = cur_total_loss
                    print("loss accleration: %6.2f | total loss: %4.2f" % (loss_acceleration, cur_total_loss))

                d = angle2vec(posvecs[i]).dot(angle2vec(negvecs[j]))
                score = (targetDots[i][j] - d)
                loss = score * score

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
