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



# Compute frequencies of words in the given sentence ws, and the current
# word frequences wfs
def update_wfs(ws, wfs):
    """ws: iterable, wfs: Counter"""
    for w in ws:
        wfs[w] += 1

# http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
# Sampler to sample from given frequency distribution
class Sampler:
    def __init__(self, wfs):

        POW = 0.75 # recommended power to raise to 
        for w in wfs:
            wfs[w] = wfs[w] ** 0.75

        # total count
        ftot = float(sum(wfs.values()))
        # words
        self.ws = list(wfs.keys())
        # word to its index
        self.ws2ix = { self.ws[i]: i for i in range(len(self.ws)) }

        # probabilities of words in the same order
        self.probs = [float(wfs[w]) / ftot for w in self.ws]

    # return words in the sampler
    def words(self):
        return self.ws

    # get the numerical index of a word
    def wordix(self, w):
        return self.ws2ix[w]

    def __len__(self):
        return len(self.ws)

    # sample N values out from the frequency distribution
    def sample(self, count=1):
        s = np.random.choice(self.ws, count, p=self.probs)
        if count == 1:
            return s[0]
        return s

# Make skipgram pairs with window size 1
def mk_skipgrams_sentence(s, sampler):
    """s: current sentence, wfs: word frequencies"""
    for i in range(1, len(s) - 1):
        yield (s[i], s[i - 1], 1)
        yield (s[i], s[i + 1], 1)
        # yield ([s[i], sampler.sample(), 0])
        # yield ([s[i], sampler.sample(), 0])

def mk_onehot(sampler, w):
    # v = torch.zeros(len(sampler)).float()
    #v[sampler.wordix(w)] = 1.0
    # return v

    return Variable(torch.LongTensor([sampler.wordix(w)]))


# Classical word2vec
# https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.ipynb
class Classical(nn.Module):
    def __init__(self, sampler, nhidden):
        self.nhidden = nhidden
        nwords = len(sampler)
        """nwords: number of words"""
        super(Classical, self).__init__()
        self.embedding = nn.Embedding(len(sampler), nhidden)

    # run the forward pass which matches word y in the context of x
    def forward(self, x_, y_):
        xembed = self.embedding(x_)
        print("xembed: %s" % (xembed, ))
        xembed = xembed.view((self.nhidden,))
        print("xembed: %s" % (xembed, ))

        yembed = self.embedding(y_)
        print("yembed: %s" % (yembed, ))
        yembed = yembed.view((1, -1))
        yembed = yembed.view((self.nhidden,))

        score = torch.dot(xembed, yembed)
        log_probs = F.logsigmoid(score)
        print("log_probs: %s" % (log_probs, ))
        return log_probs
# Corpus contains a list of sentences. Each s is a list of words
# Data pulled from:
# https://github.com/RaRe-Technologies/gensim-data
corpus = api.load('text8') 
NSENTENCES = 10
corpus = list(itertools.islice(corpus, NSENTENCES))

# Count word frequencies
wfs = Counter()
for s in corpus:
    update_wfs(s, wfs)
sampler = Sampler(wfs)

if __name__ == "__main__":
    classical = Classical(sampler, nhidden=3)
    print("network: ")
    print(classical)

    # optimise
    optimizer = optim.SGD(classical.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("criterion:\n%s" % (criterion, ))

    for s in corpus:
        # s = s[:10]
        print("training on sentence:\n---\n%s\n---" % " ".join(s))
        for train in mk_skipgrams_sentence(s, sampler):
            print("training on sample: %s" % (train,))
            (w, wctx, is_positive) = train
            x_ = mk_onehot(sampler, w)
            y_ = mk_onehot(sampler, wctx)

            optimizer.zero_grad()   # zero the gradient buffers
            y = classical(x_, y_)
            # print("y: %s" % y)
            loss = criterion(y, Variable(torch.Tensor([is_positive])))
            loss.backward()
            optimizer.step()
