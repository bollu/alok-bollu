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
import torch.nn.functional as F
import torch.optim as optim
import gensim.downloader as api
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import math
import prompt_toolkit
from prompt_toolkit.completion import WordCompleter, ThreadedCompleter
from prompt_toolkit import PromptSession
from prompt_toolkit import print_formatted_text
import progressbar
import pudb
import tabulate
from nltk.corpus import wordnet
from random import sample
import operator


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

class TimeLogger:
    """Provide an API to log start and end times of tasks"""
    def __init__(self):
        self.ts = []
        pass
    def start(self, toprint):
        depth = len(self.ts)
        self.ts.append(datetime.datetime.now())
        print(" " * 4 * depth + str(toprint) + "...")
        sys.stdout.flush()

    def log(self, toprint):
        depth = len(self.ts)
        l = "\n".join(map(lambda s: " " * 4 * depth + s, toprint.split("\n")))
        print(l)

    def end(self, toprint=None):
        if toprint is not None:
            self.log(toprint)

        depth = len(self.ts)
        start = self.ts.pop()
        now = datetime.datetime.now()
        print(" " * (4 * (depth - 1)) + "====time: %s" % (now - start))
        sys.stdout.flush()

def load_corpus(LOGGER, CORPUS_NAME):
    LOGGER.start("loading corpus: %s" % CORPUS_NAME)
    try:
        sys.path.insert(0, api.base_dir)
        module = __import__(CORPUS_NAME)
        corpus = module.load_data()
    except Exception as e:
        LOGGER.log("unable to find text8 locally.\nERROR: %s" % (e, ))
        LOGGER.start("Downloading using gensim-data...")
        corpus = api.load(CORPUS_NAME)
        LOGGER.end()
    return corpus

def preprocess_doc(LOGGER, corpus):
    """load the corpus, and pull NDOCS from the corpus if it's not None"""
    def flatten(ls):
        return [item for sublist in ls for item in sublist]

    def get_freq_cutoff(freqs, cutoff):
        """Get the frequency below which the data accounts for < cutoff
        freqs: list of frequencies
        0 <= cutoff <= 1
        """
        freqs = list(freqs)
        TOTFREQ = sum(freqs)
        freqs.sort()

        tot = 0
        i = 0
        # accumulate till where we reach cutoff
        while tot < TOTFREQ * cutoff and i < len(freqs): i += 1; tot += freqs[i]
        return freqs[i]

    # corpus = flatten(corpus)
    LOGGER.log("number of words in corpus (original): %s" % (len(corpus), ))

    # LOGGER.start("filtering stopwords")
    corpus = list(filter(lambda w: w not in STOPWORDS, corpus))
    # LOGGER.end("#words in corpus after filtering: %s" % (len(corpus), ))



    FREQ_CUTOFF = 0.2
    LOGGER.start("filtering low frequency words... (all frequencies that account for < %s percent of the dataset)" % (str(FREQ_CUTOFF * 100), ))
    w2f = mk_word_histogram(corpus, set(corpus))
    origlen = len(corpus)
    cutoff_freq = get_freq_cutoff(w2f.values(), FREQ_CUTOFF)
    corpus = list(filter(lambda w: w2f[w] > cutoff_freq, corpus))
    filtlen = len(corpus)
    LOGGER.end("filtered #%s (%s percent) words. New corpus size: %s (%s percent of original)" %
               (origlen - filtlen,
                float(origlen - filtlen) / float(origlen) * 100.0,
                filtlen,
                filtlen / float(origlen) * 100.0
               ))

    return corpus


def batch(xs, BATCHSIZE):
    ix = 0
    while ix + BATCHSIZE < len(xs):
        data = xs[ix:ix+BATCHSIZE]
        ix += BATCHSIZE
        yield data

# TODO: extract into a method of metric
def dots(vs, ws, metric):
    """Take the dot product of each element in vs with elements in ws"""
    # vs = [S1 x EMBEDSIZE]
    # ws = [S2 x EMBEDSIZE] | ws^t = [EMBEDSIZE x S2]
    # metric = [EMBEDSIZE x EMBEDSIZE]
    # vs * metric = [S1 x EMBEDSIZE]
    # vs * metric * ws^t = [S1 x EMBEDSIZE] x [EMBEDSIZE x S2] = [S1 x S2]

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


# TODO: extract into method of metric
def dot(v, w, metric):
    return dots(v.view(1, -1), w.view(1, -1), metric)


# TODO: extract into method of metric
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

# TODO: extract into method of metric
def normalize(DEVICE, vs, metric):
    # vs = [S1 x EMBEDSIZE]
    # metric = [EMBEDSIZE x EMBEDSIZE]
    # normvs = [S1 x EMBEDSIZE]
    normvs = torch.zeros(vs.size()).to(DEVICE)
    BATCHSIZE = 4096
    # with prompt_toolkit.shortcuts.ProgressBar() as pb:
    for i in (range(math.ceil(vs.size()[0] / BATCHSIZE))):
        vscur = vs[i*BATCHSIZE:(i+1)*BATCHSIZE, :]
        vslen = torch.sqrt(torch.diag(dots(vscur, vscur, metric)))
        vslen = vslen.view(-1, 1) # have one element per column which is the length
        normvs[i*BATCHSIZE:(i+1)*BATCHSIZE, :] = vscur / vslen

    # with prompt_toolkit.shortcuts.ProgressBar() as pb:
    #     for i in pb(range(math.ceil(vs.size()[0]))):
    #         vscur = vs[i, :]
    #         vslen = torch.sqrt(dot(vscur, vscur, metric))
    #         vslen = vslen.view(-1, 1) # have one element per column which is the length
    #         normvs[i, :] = vscur / vslen
    normvs.to(DEVICE)
    return normvs


def mk_symmetric_mat(DEVICE, n):
    """make a symmetric matrix of size (NxN) which can be gradient descended on"""

    # upper triangular matrix with entires one off the diagonal
    triu = torch.triu(nn.Parameter(torch.randn(n, n).to(DEVICE), requires_grad=True), diagonal=1)
    # diagonal matrix
    diag = torch.diag(nn.Parameter(torch.randn(n).to(DEVICE), requires_grad=True))
    return triu + diag + triu.t()

    # make an upper triangle, copy it to lower triangle, and then make a
    # random diagonal as well.
    tri = nn.Parameter(torch.randn((n*(n - 1)) // 2).to(DEVICE), requires_grad=True)
    mat = torch.zeros(n, n).to(DEVICE)
    mat[np.tril_indices(n, -1)] = tri
    mat[np.triu_indices(n, 1)] = tri
    mat[np.diag_indices(n)] = nn.Parameter(torch.randn(n).to(DEVICE), requires_grad=True)
    return mat

class Metric(nn.Module):
    def __init__(self):
        super(Metric, self).__init__()

    @property
    def mat(self):
        """return metric matrix of size EMBEDSIZE x EMBEDSIZE"""
        raise NotImplementedError()

class EuclidMetric(Metric):
    def __init__(self, DEVICE, embedsize):
        super(EuclidMetric, self).__init__()
        self.mat_ = nn.Parameter(torch.eye(embedsize).to(DEVICE), requires_grad=False)
    @property
    def mat(self):
        return self.mat_

class ReimannMetric(Metric):
    def __init__(self, DEVICE, embedsize):
        super(ReimannMetric, self).__init__()
        self.sqrt_ = nn.Parameter(torch.randn([embedsize, embedsize]).to(DEVICE), requires_grad=True)
    @property
    def mat(self):
        return torch.mm(self.sqrt_, self.sqrt_)

class PseudoReimannMetric(Metric):
    def __init__(self, DEVICE, embedsize):
        super(PseudoReimannMetric, self).__init__()
        self.mat_ = mk_symmetric_mat(DEVICE, embedsize).to(DEVICE)

    @property
    def mat(self):
        return self.mat_

def get_windowed_ixs(windowsize, ix, maxix):
    """Generator that returns values values of:
        (ix, min(max(0, ix + delta), maxix),
        delta = [-windowsize, -1] U [1, windowsize]
        0 <= ix <= self.windowsize
       Used to list tail elements in a skip gram dataset.
    """
    for d in range(max(ix - windowsize, 0), ix):
        yield (ix, d)
    for d in range(ix+1, min(ix + windowsize + 1, maxix)):
        yield (ix, d)
def flatten(xss):
    return [x for xs in xss for x in xs]



def get_columns(m, cols):
    """Index the cols columns of matrix m"""
    return m.t()[cols].t()

# def sample_word(vocab, w2f):
class SkipGramNegSamplingDataset(Dataset):
    def __init__(self, LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE, I2W, W2I):
        self.TEXT = TEXT

        self.VOCAB = VOCAB
        self.VOCABSIZE = VOCABSIZE
        self.WINDOWSIZE = WINDOWSIZE
        self.NNEGSAMPLES = 25

        self.I2W = I2W
        self.W2I = W2I

        self.begin_ixs = flatten([get_windowed_ixs(WINDOWSIZE, i, len(TEXT)) for i in range(WINDOWSIZE)])
        self.end_ixs = flatten([get_windowed_ixs(WINDOWSIZE, i, len(TEXT)) for i in range(len(TEXT) - WINDOWSIZE, len(TEXT))])


        self.W2F = mk_word_histogram(TEXT, VOCAB)
        # words in a sequence
        self.ws = list(self.W2F.keys())
        fs = np.array(list(self.W2F.values()))
        fs = np.power(fs, 0.75)
        # word probabilityes in the same sequence
        self.wps = fs / float(sum(fs))


    def num_tail_elements(self):
        """number of elements that are at the beginning / end
           which do not have a full WINDOWSIZE number of elements
           to the left / right
        """
        size = 0
        # ith element has i elements on the left and self.WINDOWSIZE
        # elements on the right
        for i in range(self.WINDOWSIZE):
            size += i + self.WINDOWSIZE
        assert(size == self.WINDOWSIZE * (self.WINDOWSIZE - 1) / 2 + self.WINDOWSIZE * self.WINDOWSIZE)
        return size


    def __getitem__(self, ix):
        if (ix < len(self.TEXT) * self.NNEGSAMPLES):
            # generate negative sample
            focusix = ix // self.NNEGSAMPLES;
            # sample according to the freq dist
            negsampleix = self.W2I[np.random.choice(self.ws, p=self.wps)]
            return {'ctx': negsampleix,
                    'focus': self.W2I[self.TEXT[focusix]],
                    'dot': torch.tensor(0.0)
                   }
        else:
            # generate positive samples
            ix -= len(self.TEXT * self.NNEGSAMPLES)

            if ix < len(self.begin_ixs):
                focusix, ctxix = self.begin_ixs[ix]
            elif len(self.begin_ixs) <= ix < len(self.begin_ixs) + len(self.end_ixs):
                focusix, ctxix = self.end_ixs[ix - len(self.begin_ixs)]
            else:
                ix = ix - len(self.begin_ixs) - len(self.end_ixs)
                focusix = ix // (2 * self.WINDOWSIZE)
                focusix += self.WINDOWSIZE
                deltaix = (ix % (2 * self.WINDOWSIZE)) - self.WINDOWSIZE
                ctxix = focusix + deltaix
            return {'ctx':self.W2I[self.TEXT[ctxix]],
                    'focus': self.W2I[self.TEXT[focusix]],
                    'dot': torch.tensor(1.0)
                    }

    def __len__(self):
        # we can't query the first or last value.
        # first because it has no left, last because it has no right

        # first self.WINDOWSIZE elements, have i elements on the left
        # and windowsize elements on the right.
        size = 0
        for i in range(self.WINDOWSIZE):
            size += i + self.WINDOWSIZE

        assert(size == len(self.begin_ixs))
        assert(size == len(self.end_ixs))

        # closed form:
        # \sum_{i=0}^{self.WINDOWSIZE - 1} (i + self.WINDOWSIZE)
        # = (self.WINDOWSIZE) (self.WINDOWSIZE - 1) / 2 + self.WINDOWSIZE * self.WINDOWSIZE

        return (((len(self.TEXT) - 2 * self.WINDOWSIZE) * (2 * self.WINDOWSIZE)
                + len(self.begin_ixs) + len(self.end_ixs))
                + len(self.TEXT) * self.NNEGSAMPLES)

class SkipGramOneHotDataset(Dataset):
    def __init__(self, LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE, I2W, W2I):
        self.TEXT = TEXT

        self.VOCAB = VOCAB
        self.VOCABSIZE = VOCABSIZE
        self.WINDOWSIZE = WINDOWSIZE

        self.I2W = I2W
        self.W2I = W2I

        self.begin_ixs = flatten([get_windowed_ixs(WINDOWSIZE, i, len(TEXT)) for i in range(WINDOWSIZE)])
        self.end_ixs = flatten([get_windowed_ixs(WINDOWSIZE, i, len(TEXT)) for i in range(len(TEXT) - WINDOWSIZE, len(TEXT))])


        self.W2F = mk_word_histogram(TEXT, VOCAB)


    def num_tail_elements(self):
        """number of elements that are at the beginning / end
           which do not have a full WINDOWSIZE number of elements
           to the left / right
        """
        size = 0
        # ith element has i elements on the left and self.WINDOWSIZE
        # elements on the right
        for i in range(self.WINDOWSIZE):
            size += i + self.WINDOWSIZE
        assert(size == self.WINDOWSIZE * (self.WINDOWSIZE - 1) / 2 + self.WINDOWSIZE * self.WINDOWSIZE)
        return size


    def __getitem__(self, ix):
        if ix < len(self.begin_ixs):
            focusix, ctxix = self.begin_ixs[ix]
        elif len(self.begin_ixs) <= ix < len(self.begin_ixs) + len(self.end_ixs):
            focusix, ctxix = self.end_ixs[ix - len(self.begin_ixs)]
        else:
            ix = ix - len(self.begin_ixs) - len(self.end_ixs)
            focusix = ix // (2 * self.WINDOWSIZE)
            focusix += self.WINDOWSIZE
            deltaix = (ix % (2 * self.WINDOWSIZE)) - self.WINDOWSIZE

        return {'focusonehot': bow_vec([self.TEXT[focusix]], self.W2I, self.VOCABSIZE),
                'ctxtruelabel': self.W2I[self.TEXT[focusix + deltaix]]
                }

    def __len__(self):
        # we can't query the first or last value.
        # first because it has no left, last because it has no right

        # first self.WINDOWSIZE elements, have i elements on the left
        # and windowsize elements on the right.
        size = 0
        for i in range(self.WINDOWSIZE):
            size += i + self.WINDOWSIZE

        assert(size == len(self.begin_ixs))
        assert(size == len(self.end_ixs))

        # closed form:
        # \sum_{i=0}^{self.WINDOWSIZE - 1} (i + self.WINDOWSIZE)
        # = (self.WINDOWSIZE) (self.WINDOWSIZE - 1) / 2 + self.WINDOWSIZE * self.WINDOWSIZE

        return (len(self.TEXT) - 2 * self.WINDOWSIZE) * (2 * self.WINDOWSIZE) + len(self.begin_ixs) + len(self.end_ixs)

class Word2ManSkipGramOneHot(nn.Module):
    def __init__(self, VOCABSIZE, EMBEDSIZE, LOGGER, DEVICE):
        super(Word2ManSkipGramOneHot, self).__init__()
        LOGGER.start("creating EMBEDM")
        self.EMBEDM = nn.Parameter(Variable(torch.randn(VOCABSIZE, EMBEDSIZE).to(DEVICE), requires_grad=True))
        self.NEGEMBEDM = nn.Parameter(Variable(torch.zeros(VOCABSIZE,
                                                           EMBEDSIZE).to(DEVICE),
                                      requires_grad = True))
        LOGGER.end()

    def forward(self, xs, metric):
        """
        xs are one hot vectors of the focus word. What we will return is
        [embed(xs) . embed(y) for y in ys].

        That is, the dot product of the embedding of the word with every
        other word

        xs = [BATCHSIZE x VOCABSIZE], one-hot in the VOCABSIZE dimension
        """

        # embedded vectors of the batch vectors
        # [BATCHSIZE x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [BATCHSIZE x EMBEDSIZE]
        xsembeds = torch.mm(xs, self.EMBEDM)

        # dots(BATCHSIZE x EMBEDSIZE],
        #     [VOCABSIZE x EMBEDSIZE],
        #     [EMBEDSIZE x EMBEDSIZE]) = [BATCHSIZE x VOCABSIZE]
        xsembeds_dots_embeds = dots(xsembeds, self.NEGEMBEDM, metric)
        # TODO: why is this correct? I don't geddit.
        # what in the fuck does it mean to log softmax cosine?
        # [BATCHSIZE x VOCABSIZE]
        xsembeds_dots_embeds = F.log_softmax(xsembeds_dots_embeds, dim=1)

        return xsembeds_dots_embeds

    def runtrain(self, traindata, metric, DEVICE):
        """
        run a training on a batch using the given optimizer.
        traindata: [BATCHSIZE x <data from self.dataset>]
        metric: torch.mat of dimension [EMBEDSIZE x EMBEDSIZE]
        """
        # [BATCHSIZE x VOCABSIZE]
        xs = traindata['focusonehot'].to(DEVICE)
        # [BATCHSIZE], contains target label per batch
        target_labels = traindata['ctxtruelabel'].to(DEVICE)

        # dot product of the embedding of the hidden xs vector with
        # every other hidden vector
        # xs_dots_embeds: [BATCSIZE x VOCABSIZE]
        xs_dots_embeds = self(xs, metric)

        l = F.nll_loss(xs_dots_embeds, target_labels)
        return l

    @classmethod
    def make_dataset(cls, LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE, I2W, W2I):
        """
        create a dataset from the given text
        """
        return SkipGramOneHotDataset(LOGGER, TEXT, VOCAB, VOCABSIZE,
                                     WINDOWSIZE, I2W, W2I)


class Word2ManSkipGramNegSampling(nn.Module):
    def __init__(self, VOCABSIZE, EMBEDSIZE, LOGGER, DEVICE):
        super(Word2ManSkipGramNegSampling, self).__init__()
        LOGGER.start("creating EMBEDM")
        self.EMBEDM = nn.Parameter(Variable(torch.randn(VOCABSIZE, EMBEDSIZE).to(DEVICE), requires_grad=True))
        self.NEGEMBEDM = nn.Parameter(Variable(torch.zeros(VOCABSIZE,
                                                           EMBEDSIZE).to(DEVICE),
                                               requires_grad = True))
        LOGGER.end()


    def forward(self, xs, ys, metric):
        """
        xs is the context word indeces
        ys are the context word indeces

        returns: zipWith dotproduct xs ys
        """

        xsembeds = self.EMBEDM[xs]
        ysembeds = self.NEGEMBEDM[ys]

        # dots(BATCHSIZE x EMBEDSIZE],
        #     [BATCHSIZE x EMBEDSIZE],
        #     [EMBEDSIZE x EMBEDSIZE]) = [BATCHSIZE x BATCHSIZE]
        # diag(BATCHSIZE x BATCHSIZE] = [BATCHSIZE x 1]
        xsembeds_dots_embeds = torch.diag(dots(xsembeds, ysembeds, metric))
        # return F.sigmoid(2 * xsembeds_dots_embeds - 1)
        return F.sigmoid(2 * xsembeds_dots_embeds - 1)

    def runtrain(self, traindata, metric, DEVICE):
        """
        run a training on a batch using the given optimizer.
        traindata: [BATCHSIZE x <data from self.dataset>]
        metric: torch.mat of dimension [EMBEDSIZE x EMBEDSIZE]
        """
        # [BATCHSIZE x VOCABSIZE]
        ctxs = traindata['ctx'].to(DEVICE)
        # [BATCHSIZE x VOCABSIZE]
        focuses = traindata['focus'].to(DEVICE)
        dots_targets = traindata['dot'].to(DEVICE)
        dots_nn = self(ctxs, focuses, metric)

        # print("CTXS: %s" % ctxs)
        # print("FOCUSES: %s" % focuses)
        # print("DOT TARGETS: %s" % dots_targets)

        # print("TARGETS SIZE:", dots_targets.size(), "NN SIZE:", dots_nn.size())
        # print(dots_targets)
        # print(dots_nn)

        l = F.binary_cross_entropy(dots_nn, dots_targets)
        return l

    @classmethod
    def make_dataset(cls, LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE, I2W, W2I):
        """
        create a dataset from the given text
        """
        return SkipGramNegSamplingDataset(LOGGER, TEXT, VOCAB, VOCABSIZE,
                                     WINDOWSIZE, I2W, W2I)

class CBOWDataset(Dataset):
    def __init__(self, LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE, I2W, W2I):
        self.TEXT = TEXT

        self.VOCAB = VOCAB
        self.VOCABSIZE = VOCABSIZE
        self.WINDOWSIZE = WINDOWSIZE

        self.I2W = I2W
        self.W2I = W2I

        LOGGER.start("counting frequency of words")
        self.W2F = mk_word_histogram(TEXT, VOCAB)
        LOGGER.end()

    def __getitem__(self, focusix):
        ctxws = [self.TEXT[focusix + d] for d in range(-self.WINDOWSIZE, self.WINDOWSIZE + 1)
                 if d != 0 and 0 <= focusix + d < len(self.TEXT)]

        # given context, number of words in context, produce word at focus
        return {'ctxhot': bow_avg_vec(ctxws, self.W2I, self.VOCABSIZE), #[1xVOCAB]
                'focustruelabel': torch.tensor(self.W2I[self.TEXT[focusix]]) # [1]
                }

    def __len__(self):
        # we can't query the first or last value.
        # first because it has no left, last because it has no right
        return len(self.TEXT)

class Word2ManCBOW(nn.Module):
    def __init__(self, VOCABSIZE, EMBEDSIZE, DEVICE):
        super(Word2ManCBOW, self).__init__()
        self.EMBEDM = nn.Parameter(Variable(torch.randn(VOCABSIZE, EMBEDSIZE).to(DEVICE), requires_grad=True))
        # TODO: I have so. many. questions
        # 1. Why linear?
        self.embed2vocab = nn.Parameter(Variable(torch.randn(EMBEDSIZE, VOCABSIZE).to(DEVICE), requires_grad=True))

        for p in self.parameters():
            print("PARAMETER dim: %s" % (p.size(), ))
        print(self)


    def forward(self, xs, metric):
        """
        xs = [BATCHSIZE x VOCABSIZE], **average n-hot** in the VOCABSIZE dimension
        """
        # Step 0. find average hidden vector, from input *average* n-hot
        # vector.
        # embedded vectors of the batch vectors. Will provide linear
        # combination of context vectors
        # [BATCHSIZE x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [BATCHSIZE x EMBEDSIZE]
        # [1 0 1, 0 1 0] -> [e1 + e3, e2]
        xsembeds = torch.mm(xs, self.EMBEDM)


        # [BATCHSIZE x EMBEDSIZE] x [EMBEDSIZE x VOCABSIZE] = [BATCHSIZE x VOCABSIZE]
        xsouts = torch.mm(xsembeds, self.embed2vocab)

        # Step 3. softmax to convert to probability distribution
        # TODO: why is this correct? I don't geddit.
        # what in the fuck does it mean to log softmax the mean of hidden vectors
        # sent out to an output dimension?
        # [BATCHSIZE x VOCABSIZE]
        xsouts = F.log_softmax(xsouts, dim=1)
        return xsouts

    def runtrain(self, traindata, metric, DEVICE):
        """
        run a training on a batch using the given optimizer.
        traindata: [BATCHSIZE x <data from self.dataset>]
        metric: torch.mat of dimension [EMBEDSIZE x EMBEDSIZE]
        """
        # [BATCHSIZE x VOCABSIZE]
        xs = traindata['ctxhot'].to(DEVICE)
        # [BATCHSIZE], contains target label per training sample in batch
        focustruelabel = traindata['focustruelabel'].to(DEVICE)

        # outs: [BATCHSIZE x VOCABSIZE]
        outs = self(xs, metric)

        l = F.nll_loss(outs, focustruelabel)
        return l

    @classmethod
    def make_dataset(cls, LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE):
        """
        create a dataset from the given text
        """
        return CBOWDataset(LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE)

class Parameters:

    def __init__(self, LOGGER, DEVICE, corpus, NDOCS, EPOCHS, BATCHSIZE, EMBEDSIZE, LEARNING_RATE,
                   WINDOWSIZE, create_time, metrictype, traintype,
                          metric_state_dict=None,
                          word2man_state_dict=None,
                          optimizer_state_dict=None):
        assert ((metric_state_dict is None and word2man_state_dict is None
                and optimizer_state_dict is None) or
                (metric_state_dict is not None and word2man_state_dict is not None
                 and optimizer_state_dict is not None))

        self.corpus = corpus
        self.NDOCS = int(NDOCS)
        self.DEVICE = DEVICE
        self.EPOCHS = int(EPOCHS)
        self.BATCHSIZE = int(BATCHSIZE)
        self.EMBEDSIZE = int(EMBEDSIZE)
        self.LEARNING_RATE = float(LEARNING_RATE)
        self.WINDOWSIZE = int(WINDOWSIZE)
        self.create_time = create_time
        self.metrictype = metrictype
        self.traintype = traintype

        if (self.NDOCS is not None):
            print("NDOCS:", self.NDOCS)
            self.corpus = list(self.corpus)[:self.NDOCS]

        self.TEXT = [preprocess_doc(LOGGER, doc) for doc in self.corpus]


        LOGGER.start("building vocabulary")
        self.VOCAB = set(flatten(self.TEXT))
        self.VOCABSIZE = len(self.VOCAB)
        self.I2W = dict(enumerate(self.VOCAB))
        self.W2I = { v: k for (k, v) in self.I2W.items() }
        LOGGER.end()
        LOGGER.start("creating metric")

        print("METRIC:", metrictype)
        if metrictype == "euclid":
            self.METRIC = EuclidMetric(self.DEVICE, self.EMBEDSIZE)
        elif metrictype == "reimann":
            self.METRIC = ReimannMetric(self.DEVICE, self.EMBEDSIZE)
        else:
            assert(metrictype == "pseudoreimann")
            self.METRIC = PseudoReimannMetric(self.DEVICE, self.EMBEDSIZE)
        # check that metric is subclass of Metric
        assert(self.METRIC is not None)
        # TODO: check that metric is a subclass
        if metric_state_dict is not None:
            self.METRIC.load_state_dict(metric_state_dict, strict=True)
        LOGGER.end()

        LOGGER.start("creating word2man")
        if traintype == "skipgramonehot":
            self.WORD2MAN = Word2ManSkipGramOneHot(self.VOCABSIZE,
                                                   self.EMBEDSIZE, LOGGER, self.DEVICE)
        elif traintype == "skipgramnegsampling":
            self.WORD2MAN = Word2ManSkipGramNegSampling(self.VOCABSIZE,
                                                   self.EMBEDSIZE, LOGGER, self.DEVICE)
        elif traintype == "skipgramnhot":
            raise RuntimeError("unimplemented n-hot skipgram")
        else:
            assert(traintype == "cbow")
            self.WORD2MAN = Word2ManCBOW(self.VOCABSIZE, self.EMBEDSIZE, self.DEVICE)

        if word2man_state_dict is not None:
            self.WORD2MAN.load_state_dict(word2man_state_dict, strict=True)
            self.WORD2MAN.train()
        LOGGER.end()

        LOGGER.start("creating OPTIMISER")
        self.optimizer = optim.SGD(itertools.chain(self.WORD2MAN.parameters(), self.METRIC.parameters()), lr=self.LEARNING_RATE)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        LOGGER.end()


        LOGGER.start("creating dataset...")
        self.DATASET = ConcatDataset([self.WORD2MAN.make_dataset(LOGGER,
                                                                 doc,
                                                                 self.VOCAB,
                                                                 self.VOCABSIZE,
                                                                 self.WINDOWSIZE,
                                                                 self.I2W,
                                                                 self.W2I) for doc in self.TEXT])
        LOGGER.end()

        # TODO: pytorch dataloader is sad since it doesn't save state.
        # make a version that does save state.
        LOGGER.start("creating DATA")
        self.DATALOADER = DataLoader(self.DATASET,
                                     batch_size=self.BATCHSIZE,
                                     shuffle=True)
        LOGGER.end()

    def get_model_state_dict(self):
        st =  {
            "NDOCS": self.NDOCS,
            "EPOCHS": self.EPOCHS,
            "BATCHSIZE": self.BATCHSIZE,
            "EMBEDSIZE": self.EMBEDSIZE,
            "LEARNINGRATE": self.LEARNING_RATE,
            "WINDOWSIZE": self.WINDOWSIZE,
            "CREATE_TIME": self.create_time,
            "METRICTYPE": self.metrictype,
            "TRAINTYPE": self.traintype,
            "WORD2MAN": self.WORD2MAN.state_dict(),
            "METRIC": self.METRIC.state_dict(),
            "OPTIMIZER": self.optimizer.state_dict()
        }
        return st

    @classmethod
    def load_model_state_dict(self, LOGGER, DEVICE, corpus, state):
        NDOCS = state["NDOCS"]
        EPOCHS = state["EPOCHS"]
        BATCHSIZE = state["BATCHSIZE"]
        EMBEDSIZE = state["EMBEDSIZE"]
        LEARNING_RATE = state["LEARNINGRATE"]
        WINDOWSIZE = state["WINDOWSIZE"]
        create_time = state["CREATE_TIME"]
        metrictype = state["METRICTYPE"]
        traintype = state["TRAINTYPE"]

        return Parameters(LOGGER,
                          DEVICE,
                          corpus=corpus,
                          NDOCS=NDOCS,
                          EPOCHS=EPOCHS,
                          BATCHSIZE=BATCHSIZE,
                          EMBEDSIZE=EMBEDSIZE,
                          LEARNING_RATE=LEARNING_RATE,
                          WINDOWSIZE=WINDOWSIZE,
                          create_time=create_time,
                          metrictype=metrictype,
                          traintype=traintype,
                          metric_state_dict=state["METRIC"],
                          word2man_state_dict=state["WORD2MAN"],
                          optimizer_state_dict=state["OPTIMIZER"])

def mk_word_histogram(ws, vocab):
    """count frequency of words in words, given vocabulary size."""
    w2f = { w : 0 for w in vocab }

    for w in ws:
        w2f[w] += 1
    return w2f

def bow_vec(ws, W2I, VOCABSIZE):
    """
    bag of words vector corresponding to words in ws
    """
    v = Variable(torch.zeros(VOCABSIZE).float())
    for w in ws: v[W2I[w]] = 1.0
    return v


def bow_avg_vec(ws, W2I, VOCABSIZE):
    """
    bag of words *average* vector corresponding to words in ws.
    That is, total sum of weights is 1
    """
    N = float(len(ws))
    v = Variable(torch.zeros(VOCABSIZE).float())
    for w in ws: v[W2I[w]] += 1.0 / N
    return v

def dump_sage(PARAMS):
    embednp = PARAMS.WORD2MAN.EMBEDM.detach().numpy()
    metricnp = PARAMS.METRIC.mat.detach().numpy()

    np.savez_compressed("NPSAVEFILE.npz", embed=embednp, metric=metricnp)


def find_close_vectors(PARAMS, DEVICE, EMBEDNORM, v):
    """ Find vectors close to w in the normalized embedding"""
    # dot [1 x EMBEDSIZE] [VOCABSIZE x EMBEDSIZE] = [1 x VOCABSIZE]
    v = normalize(DEVICE, v, PARAMS.METRIC.mat)
    wix2sim = dots(v, EMBEDNORM, PARAMS.METRIC.mat)

    wordweights = [(PARAMS.I2W[i], wix2sim[0][i].item()) for i in range(PARAMS.VOCABSIZE)]
    # The story of floats which cannot be ordered. I found
    # out I need this filter once I found this entry in the
    # table: ('classified', nan)
    wordweights = list(filter(lambda ww: not math.isnan(ww[1]), wordweights))
    # sort AFTER removing NaNs
    wordweights.sort(key=lambda wdot: wdot[1], reverse=True)
    return wordweights


def word_to_embed_vector(PARAMS, w):
    """
    find the embedded vector of word w
    returns: [1 x EMBEDSIZE]
    """
    v = PARAMS.WORD2MAN.EMBEDM[PARAMS.W2I[w], :]
    return v.view(1, -1)


def replcli(PARAMS, LOGGER, DEVICE):
    """Call to launch prompt interface."""

    LOGGER.start("normalizing EMBEDM...")
    PARAMS.WORD2MAN.EMBEDM = PARAMS.WORD2MAN.EMBEDM.to(DEVICE)
    # PARAMS.METRIC.mat = PARAMS.METRIC.mat.to(DEVICE)
    EMBEDNORM = normalize(DEVICE, PARAMS.WORD2MAN.EMBEDM, PARAMS.METRIC.mat).to(DEVICE)
    LOGGER.end("done.")


    def prompt_word(session):
        """Prompt for a word and print the closest vectors to the word"""
        # [VOCABSIZE x EMBEDSIZE]
        COMPLETER = ThreadedCompleter(WordCompleter(PARAMS.VOCAB))

        raw = session.prompt("type in command>", completer=COMPLETER).split()
        # raw = session.prompt("type in command>", completer=COMPLETER).split()
        if len(raw) == 0:
            return
        if raw[0] == "help" or raw[0] == "?":
            print_formatted_text("near <word> | sim <w-1> <w2> <w3> | dot <w1> <w2> | metric | debug")
            return
        elif raw[0] == "debug":
            pudb.set_trace()
        elif raw[0] == "near" or raw[0] == "neighbour":
            if len(raw) != 2:
                print_formatted_text("error: expected near <w>")
                return
            wordweights = find_close_vectors(PARAMS, DEVICE, EMBEDNORM, word_to_embed_vector(PARAMS, raw[1]))
            for (word, weight) in wordweights[:15]:
                print_formatted_text("\t%40s %0.2f" % (word, weight))
        elif raw[0] == "sim":
            if len(raw) != 4:
                print_formatted_text("error: expected sim <w1> <w2> <w3>")
                return
            v1 = word_to_embed_vector(PARAMS, raw[1])
            v2 = word_to_embed_vector(PARAMS, raw[2])
            v3 = word_to_embed_vector(PARAMS, raw[3])
            vsim = normalize(DEVICE, v2 - v1+ v3, PARAMS.METRIC.mat)
            wordweights = find_close_vectors(PARAMS, DEVICE, EMBEDNORM, vsim)
            for (word, weight) in wordweights[:15]:
                print_formatted_text("\tnormal(b - a + c) %s: %s" % (word, weight))


            v1 = normalize(DEVICE, word_to_embed_vector(PARAMS, raw[1]), PARAMS.METRIC.mat)
            v2 = normalize(DEVICE, word_to_embed_vector(PARAMS, raw[2]), PARAMS.METRIC.mat)
            v3 = normalize(DEVICE, word_to_embed_vector(PARAMS, raw[3]), PARAMS.METRIC.mat)

            vsim = normalize(DEVICE, v2 - v1 + v3, PARAMS.METRIC.mat)
            wordweights = find_close_vectors(PARAMS, DEVICE, EMBEDNORM, vsim)
            for (word, weight) in wordweights[:15]:
                print_formatted_text("\tnormal(normal(b) - normal(a) + normal(c)): %s: %s" % (word, weight))
            wordweights = find_close_vectors(PARAMS, DEVICE, EMBEDNORM, vsim)
            for (word, weight) in wordweights[:15]:
                print_formatted_text("\tnormal(normal(b) - normal(a) + normal(c)): %s: %s" % (word, weight))

        elif raw[0] == "dot":
            if len(raw) != 3:
                print_formatted_text("error: expected dot <w1> <w2>")
                return

            v1 = normalize(DEVICE, word_to_embed_vector(PARAMS, raw[1]), PARAMS.METRIC.mat)
            v2 = normalize(DEVICE, word_to_embed_vector(PARAMS, raw[2]), PARAMS.METRIC.mat)
            print_formatted_text("\t%s" % (dots(v1, v2, PARAMS.METRIC.mat), ))
        elif raw[0] == "metric":
            print_formatted_text(PARAMS.METRIC.mat)
            evals,evecs = torch.eig(PARAMS.METRIC.mat,eigenvectors=True)

            evals = [evals[i][0].item() for i in range(evals.size()[0])]
            evals.sort(key=abs, reverse=True)
        else:
            print_formatted_text("invalid command, type ? for help")

    session = PromptSession()
    with torch.no_grad():
        # event loop
        while True:
            try:
                prompt_word(session)
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except KeyError as e:
                print_formatted_text("exception:\n%s" % (e, ))


# @EXPERIMENT.capture
def traincli(savepath, savetimesecs, PARAMS, LOGGER, DEVICE):
    def save():
        if savepath is None: return
        LOGGER.start("\nsaving model to: %s" % (savepath))
        with open(savepath, "wb") as sf:
            state_dict =  PARAMS.get_model_state_dict()
            torch.save(state_dict, sf)


    # Notice that we do not normalize the vectors in the hidden layer
    # when we train them! this is intentional: In general, these vectors don't
    # seem to be normalized by most people, so it's weird if we begin to
    # normalize them.
    # Read also: what is the meaning of the length of a vector in word2vec?
    # https://stackoverflow.com/questions/36034454/what-meaning-does-the-length-of-a-word2vec-vector-have
    loss_sum = 0
    ix = 0
    time_last_save = datetime.datetime.now()
    time_last_print = datetime.datetime.now()
    tbegin = datetime.datetime.now()
    last_print_ix = 0
    for epoch in range(PARAMS.EPOCHS):
        for traindata in PARAMS.DATALOADER:
            ix += 1
            PARAMS.optimizer.zero_grad()   # zero the gradient buffers
            l = PARAMS.WORD2MAN.runtrain(traindata, PARAMS.METRIC.mat, DEVICE)
            loss_sum += l.item()
            l.backward()
            PARAMS.optimizer.step()
            # updating data
            now = datetime.datetime.now()
            # num units / unit time
            ratesec = (now - tbegin) / ix
            # time left
            tleft = ratesec * (len(PARAMS.DATALOADER) * PARAMS.EPOCHS - ix)


            # printing
            TARGET_PRINT_TIME_IN_S = 1
            if (now - time_last_print).seconds >= TARGET_PRINT_TIME_IN_S:
                nbatches = ix - last_print_ix
                print("LOSSES sum: %0.2f | avg per batch(#batch=%s): %0.2f | avg per elements(#elems=%s): %0.2f | eta: %s" %
                      (loss_sum,
                       nbatches,
                       loss_sum / nbatches,
                       nbatches * PARAMS.BATCHSIZE,
                       loss_sum / (nbatches * PARAMS.BATCHSIZE),
                      tleft))
                loss_sum = 0
                time_last_print = now
                last_print_ix = ix

            # saving
            TARGET_SAVE_TIME_IN_S = savetimesecs # save every X minutes
            if (now - time_last_save).seconds > TARGET_SAVE_TIME_IN_S:
                save()
                time_last_save = now
    save()


def wordnet_evaluate(PARAMS, LOGGER, DEVICE):

    # PARAMS.DATASET.VOCAB
    # PARAMS.DATASET.TEXT
    # word_to_embed_vector
    # find_close_vectors
    # dots -> dot products
    # cosinesim -> cosine similarity

    # Get a sample of unique words (VOCAB)
    # For each word in the vocab, find the 10 closest vectors
    # Find these words
    # Find wup_similarity and path_similarity of the words in the list

    word_sym_pairs = []

    sampled_words = sample(PARAMS.VOCAB, 10)
    for word in sampled_words:
        similar_words = find_close_vectors(word_to_embed_vector(PARAMS, word))[:10]
        word_sym_pairs.append((word, similar_words))

    # Approximation made here for now: Only the first sense of each similar word has been taken
    # Later task: Will compare similarities from the entire synset.

    print("=================>OUR METRIC WORDS<===========================")

    for (word, sim_words) in word_sym_pairs:
            wn_word = wordnet.synsets(word)
            rows = []
            for (sim_word, score) in sim_words:
                if len(wordnet.synsets(sim_word)) > 0:
                    try:
                        wn_sim = wordnet.synsets(sim_word)
                        wup = [wordnet.wup_similarity(sense, sim_sense) for sense in wn_word for sim_sense in wn_sim]
                        path = [wordnet.path_similarity(sense, sim_sense) for sense in wn_word for sim_sense in wn_sim]
                        wup = [0.0 if w==None else w for w in wup]
                        path = [0.0 if w==None else w for w in path]

                        if wup==[]:
                            rows.append([word,sim_word, 0, 0, score])
                        else:
                            rows.append([word, sim_word, max(wup), max(path), score])
                    #    print(word + "\t" + sim_word + "\t" + str(wup_sim) + "\t" + str(path_sim) + "\t" + str(score))
                    except IndexError:
                        print("%word not in wordnet: %s" % sim_word)
            print(tabulate.tabulate(rows, headers=["Word", "Similar", "WUP Score", "Path Score", "Our Product"]))

    # What are the closest words according to wordnet that belong in our corpus?
    print("=================>WORDNET SIMILARITY WORDS<===========================")

    for word in sampled_words:
        rows = []
        for corpus_word in PARAMS.DATASET.VOCAB:
            wn_word = wordnet.synsets(word)
            wn_corpus_word = wordnet.synsets(corpus_word)
            wup = [wordnet.wup_similarity(wn, wn_corpus) for wn in wn_word for wn_corpus in wn_corpus_word]
            path = [wordnet.path_similarity(wn, wn_corpus) for wn in wn_word for wn_corpus in wn_corpus_word]
            wup = [0.0 if w==None else w for w in wup]
            path = [0.0 if w==None else w for w in path]

            # print(wup[:10])

            if wup == []:
                rows.append([word, corpus_word, 0.0, 0.0, cosinesim(word_to_embed_vector(word), word_to_embed_vector(corpus_word), PARAMS.METRIC.mat)])
            else:
                rows.append([word, corpus_word, max(wup), max(path), cosinesim(word_to_embed_vector(word), word_to_embed_vector(corpus_word), PARAMS.METRIC.mat)])

        rows.sort(key=operator.itemgetter(2), reverse=True)
        rows = rows[:10]
        print(tabulate.tabulate(rows, headers=["Word", "Similar", "WUP Score", "Path Score", "Our Product"]))

def current_time_str():
    """Return the current time as a string"""
    return datetime.datetime.now().strftime("%X-%a-%b")

def parse(s):
    def DEFAULT_MODELPATH():
        return "save-auto-%s.model" % (current_time_str(), )
    p = argparse.ArgumentParser()

    sub = p.add_subparsers(dest="command")
    train = sub.add_parser("train", help="train the model")
    train.add_argument("--loadpath", help="path to model file to load from", default=None)
    train.add_argument("--savepath", help="path to save model to", default=DEFAULT_MODELPATH())
    train.add_argument("--epochs", default=5)
    train.add_argument("--batchsize", default=64)
    train.add_argument("--embedsize", default=200)
    train.add_argument("--learningrate", default=0.05)
    train.add_argument("--windowsize", default=4)
    # number of documents to process. is None by default to run on the
    # entire corpus
    train.add_argument("--ndocs", default=None)
    train.add_argument("--metrictype", help="type of metric to use",
                       choices=["euclid", "reimann", "pseudoreimann"])
    train.add_argument("--traintype", help="training method to use",
                       choices=["cbow", "skipgramonehot",
                                "skipgramnegsampling", "skipgramnhot"])
    train.add_argument("--savetimesecs", help="number of seconds to be elapsed before saving", default=5*60)

    evalrepl = sub.add_parser("evalrepl", help="evaluate the model in a repl")
    evalrepl.add_argument("loadpath", help="path to model file to load from", default=None)


    testrepl = sub.add_parser("testrepl", help="interactively test the model")


    dumpsage = sub.add_parser("dumpsage", help="dump data to be imported into sage")
    dumpsage.add_argument("loadpath", help="path to model file to load from", default=None)


    return p.parse_args(s)


def main():
    LOGGER = TimeLogger()

    # parse args
    PARSED = parse(sys.argv[1:])
    assert (PARSED.command in ["train", "wordneteval", "testrepl", "evalrepl", "dumpsage"])

    # setup device
    LOGGER.start("setting up device")
    DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')
    LOGGER.end("device: %s" % DEVICE)


    # load corpus
    # TODO: fold corpus state into PARAMS
    corpus = load_corpus(LOGGER, "text8")

    # if we are testing, let the corpus be a synthetic corpus
    if PARSED.command == "testrepl":
        PARAMS = Parameters(LOGGER,
                            DEVICE,
                            corpus=corpus,
                            NDOCS=1,
                            EPOCHS=15,
                            BATCHSIZE=32,
                            EMBEDSIZE=3,
                            LEARNING_RATE=0.025,
                            WINDOWSIZE=2,
                            create_time=current_time_str(),
                            metrictype="euclid",
                            traintype="skipgramnegsampling")
    # if we are evaluating, just load data
    elif PARSED.command == "wordneteval" or PARSED.command == "evalrepl":
        state = torch.load(PARSED.loadpath, map_location=DEVICE)
        PARAMS = Parameters.load_model_state_dict(LOGGER,
                                                  DEVICE,
                                                  corpus,
                                                  state)
    # if we are training and the load path exists, load
    else:
        assert(PARSED.command == "train")

        if PARSED.loadpath is not None and os.path.exists(PARSED.loadpath):
            LOGGER.start("loaded params from: %s" % PARAMS.create_time)
            state = torch.load(PARSED.loadpath, map_location=DEVICE)
            PARAMS = Parameters.load_model_state_dict(LOGGER,
                                                      DEVICE,
                                                      corpus,
                                                      state)
            LOGGER.end()
        else:
            PARAMS = Parameters(LOGGER,
                                DEVICE,
                                corpus,
                                NDOCS=PARSED.ndocs,
                                EPOCHS=PARSED.epochs,
                                BATCHSIZE=PARSED.batchsize,
                                EMBEDSIZE=PARSED.embedsize,
                                LEARNING_RATE=PARSED.learningrate,
                                WINDOWSIZE=PARSED.windowsize,
                                create_time=current_time_str(),
                                metrictype=PARSED.metrictype,
                                traintype=PARSED.traintype)

    assert(PARAMS is not None)

    if PARSED.command == "train":
        traincli(PARSED.savepath, PARSED.savetimesecs, PARAMS, LOGGER, DEVICE)
    elif PARSED.command == "dumpsage":
        dump_sage(PARAMS)
    elif PARSED.command == "evalrepl":
        replcli(PARAMS, LOGGER, DEVICE)
    elif PARSED.command == "testrepl":
        SECS_TO_SAVE = 30
        traincli('testsave', SECS_TO_SAVE, PARAMS, LOGGER, DEVICE)
        replcli(PARAMS, LOGGER, DEVICE)
    elif PARSED.command == "wordneteval":
        wordnet_evaluate(PARAMS, LOGGER, DEVICE)
        replcli(PARAMS, LOGGER, DEVICE)
    else:
        raise RuntimeError("unknown command: %s" % PARSED.command)

if __name__ == "__main__":
    main()