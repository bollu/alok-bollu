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


torch.manual_seed(0)

STEPSIZE = 0.1
LENGTH_WEIGHT = 0.0
EMBEDSIZE = 2
NVECS = 10
NNEGSAMPLES = 15
STARTING_ALPHA = 0.001
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')

posvecs = torch.randn([NVECS, EMBEDSIZE, EMBEDSIZE], device=DEVICE, requires_grad=True, dtype=torch.float)
targetDots = torch.randn([NVECS, NVECS], device=DEVICE, requires_grad=False, dtype=torch.float)

def normalize(v):
    return v / v.det()


def calculate_loss(pos):
    l = 0
    with torch.no_grad():
        for i in range (NVECS):
            for j in range (NVECS):
                score = (targetDots[i][j] - 
                         normalize(pos[i].T).mul(normalize(pos[j])).trace())
                l += score * score
    return l

def propose():
    global STEPSIZE
    proposal_posvecs = posvecs + \
        STEPSIZE * torch.randn([NVECS, EMBEDSIZE, EMBEDSIZE], device=DEVICE,
                               requires_grad=True)
    # proposal_negvecs = negvecs + \
    #     STEPSIZE * torch.randn([NVECS, EMBEDSIZE-1], device=DEVICE,
    #                            requires_grad=True)
    # return (proposal_posvecs, proposal_negvecs)
    return proposal_posvecs


def mhStep():
    """returns whether accepted proposal"""
    global posvecs
    global STEPSIZE
    cur_loss = calculate_loss(posvecs)

    # proposal_posvecs, proposal_negvecs = propose()
    proposal_posvecs = propose()
    proposal_loss = calculate_loss(proposal_posvecs)

    r = torch.empty(1, 1, device=DEVICE).uniform_(0, 1)

    # if proposal_loss < cur_loss, cur_loss / proposal_loss > 1
    # 2^(cur_loss - proposal_loss)
    if r < 2 ** (cur_loss - proposal_loss):
        posvecs = proposal_posvecs
        # negvecs = proposal_negvecs
        accepted = True
    else:
        accepted = False



    print("%s | stepsize: %4.6ff | r: %4.6f <  improvement: %4.6f" %
            ("ACCEPT" if accepted else "REJECT", STEPSIZE, r, cur_loss / proposal_loss))
    return accepted

def gradientStep():
    loss = 0
    posvecs.detach_()
    posvecs.requires_grad_()
    for i in range(NVECS):
        for j in range(NVECS):
            if i == j: continue
            d = posvecs[i].T.mul(posvecs[j]).trace()
            score = (targetDots[i][j] - d)
            # loss = score * score + \
            #         LENGTH_WEIGHT * (1.0 -
            #                          posvecs[i].T.mul(posvecs[i]).trace()) + \
            #         LENGTH_WEIGHT * (1.0 -
            #                          posvecs[j].T.mul(posvecs[j]).trace())
            loss += score * score

    grad = loss.backward()
    posvecs.data.sub_(posvecs.grad.data * STARTING_ALPHA)
    posvecs.grad.data.zero_()

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
        gradientStep()

        if nvecs - last_nvecs >= 10:
            last_nvecs = nvecs
            cur_total_loss = calculate_loss(posvecs)
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
