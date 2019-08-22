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
STEPSIZE = 0.1
EMBEDSIZE = 3
NVECS = 10
NNEGSAMPLES = 15
STARTING_ALPHA = 0.1
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')

posvecs = torch.randn([NVECS, EMBEDSIZE-1], device=DEVICE, requires_grad=False, dtype=torch.float)
negvecs = torch.randn([NVECS, EMBEDSIZE-1], device=DEVICE, requires_grad=False, dtype=torch.float)
targetDots = torch.randn([NVECS, NVECS], device=DEVICE, requires_grad=False, dtype=torch.float)

torch.autograd.set_detect_anomaly(DEBUG)


def normalize(v):
    return v / torch.sqrt(v.dot(v))


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

def calculate_loss(pos, neg):
    l = 0
    with torch.no_grad():
        for i in range (NVECS):
            for j in range (NVECS):
                score = (targetDots[i][j] - angle2vec(pos[i]).dot(angle2vec(neg[j])))
                # score = (targetDots[i][j] - normalize(pos[i]).dot(normalize(neg[j])))
                # score = targetDots[i][j] - pos[i].dot(neg[j])
                l += score * score
    return l

def propose():
    global STEPSIZE
    proposal_posvecs = posvecs + STEPSIZE * torch.randn([NVECS, EMBEDSIZE-1], device=DEVICE)
    proposal_negvecs = negvecs + STEPSIZE * torch.randn([NVECS, EMBEDSIZE-1], device=DEVICE)
    return (proposal_posvecs, proposal_negvecs)


def mhStep():
    """returns whether accepted proposal"""
    global posvecs
    global negvecs
    global STEPSIZE
    cur_loss = calculate_loss(posvecs, negvecs)

    proposal_posvecs, proposal_negvecs = propose()
    proposal_loss = calculate_loss(proposal_posvecs, proposal_negvecs)

    r = torch.empty(1, 1, device=DEVICE).uniform_(0, 1)

    # if proposal_loss < cur_loss, cur_loss / proposal_loss > 1
    # 2^(cur_loss - proposal_loss)
    if r < 2 ** (cur_loss - proposal_loss):
        posvecs = proposal_posvecs
        negvecs = proposal_negvecs
        accepted = True
    else:
        accepted = False


    print("%s | stepsize: %4.6ff | r: %4.6f <  improvement: %4.6f" %
            ("ACCEPT" if accepted else "REJECT", STEPSIZE, r, cur_loss / proposal_loss))
    return accepted


def trainW2V():
    prev_total_loss = 0
    loss_acceleration = 1
    last_nvecs = 0
    nvecs = 0
    nproposed = 0
    nproposal_accepted = 0


    while abs(loss_acceleration) > 1e-2:
        alpha = STARTING_ALPHA
        nvecs += 1

        nproposed += 1
        nproposal_accepted += 1 if mhStep() else 0

        if nvecs - last_nvecs >= 10:
            last_nvecs = nvecs
            cur_total_loss = calculate_loss(posvecs, negvecs)
            loss_acceleration = prev_total_loss - cur_total_loss
            prev_total_loss = cur_total_loss
            print("acceptance ratio: %6.2f | loss accleration: %6.2f | total loss: %4.2f" % 
                  (nproposal_accepted / nproposed, loss_acceleration, cur_total_loss))





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
