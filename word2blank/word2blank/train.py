#!/usr/bin/env python
from collections import Counter
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import numpy as np
# https://github.com/RaRe-Technologies/gensim-data


def sample_list(s):
    i = torch.randint(0, len(s), (1,))
    return s[i]

# Compute frequencies of words in the given sentence ws, and the current
# word frequences wfs
def update_wfs(ws, wfs):
    """ws: iterable, wfs: Counter"""
    for w in ws:
        wfs[w] += 1

# http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
class Sampler:
    def __init__(self, wfs):

        POW = 0.75 # recommended power to raise to 
        for w in wfs:
            wfs[w] = wfs[w] ** 0.75

        # total count
        ftot = float(sum(wfs.values()))
        # words
        self.ws = wfs.keys()
        # probabilities of words in the same order
        self.probs = [float(wfs[w]) / ftot for w in self.ws]

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
        # should I generate a correct word or not?
        pick_correct_word = torch.randint(0, 2, (1,))

        if pick_correct_word:
                yield ([s[i], s[i - 1], 1]),
                yield ([s[i], s[i + 1], 1])
        else:
            yield ([s[i], sampler.sample(), 0])

# Corpus contains a list of sentences. Each s is a list of words
corpus = api.load('text8')  # download the corpus and return it opened as an iterable

# Count word frequencies
wfs = Counter()
for s in itertools.islice(corpus, 10):
    update_wfs(s, wfs)

sampler = Sampler(wfs)

sc = Counter(sampler.sample(1000))

skipgrams = mk_skipgrams_sentence(list(itertools.islice(corpus, 1))[0], sampler)
