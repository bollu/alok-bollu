#!/usr/bin/env python3
from collections import Counter


import argparse
import sys
import datetime

def current_time_str():
    """Return the current time as a string"""
    return datetime.datetime.now().strftime("%X-%a-%b")

def parse(s):
    def DEFAULT_MODELPATH():
        now = datetime.datetime.now()
        return "save-auto-%s.model" % (current_time_str(), )
    p = argparse.ArgumentParser()

    sub = p.add_subparsers(dest="command")
    train = sub.add_parser("train", help="train the model")
    train.add_argument("--loadpath", help="path to model file to load from", default=None)
    train.add_argument("--savepath", help="path to save model to", default=DEFAULT_MODELPATH())
    train.add_argument("--metrictype", help="type of metric to use",
                       choices=["euclid", "reimann", "pseudoreimann"])
    train.add_argument("--traintype", help="training method to use",
                       choices=["cbow", "skipgramonehot", "skipgramnhot"])

    test = sub.add_parser("test", help="test the model")
    test.add_argument("loadpath",  help="path to model file")

    evaluate = sub.add_parser("eval", help="evaluate the model")
    evaluate.add_argument("loadpath", help="path to model file")

    return p.parse_args(s)
# if launching from the shell, parse first, then start loading datasets...
if __name__ == "__main__":
    global PARSED
    PARSED = parse(sys.argv[1:])
    assert (PARSED.command in ["train", "test", "eval"])

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
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit import print_formatted_text
import prompt_toolkit.shortcuts
import sacred
import sacred.observers
import progressbar
import pudb
import numpy as np
import tabulate


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

def load_corpus(LOGGER, nwords):
    """load the corpus, and pull nwords from the corpus if it's not None"""
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

    CORPUS_NAME = "text8"
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

    corpus = list(corpus)
    corpus = flatten(corpus)
    LOGGER.log("number of words in corpus (original): %s" % (len(corpus), ))

    LOGGER.start("filtering stopwords")
    corpus = list(filter(lambda w: w not in STOPWORDS, corpus))
    LOGGER.end("#words in corpus after filtering: %s" % (len(corpus), ))



    LOGGER.start("filtering low frequency words... (all frequencies that account for < 20% of the dataset)")
    w2f = mk_word_histogram(corpus, set(corpus))
    origlen = len(corpus)
    cutoff_freq = get_freq_cutoff(w2f.values(), 0.2)
    corpus = list(filter(lambda w: w2f[w] > cutoff_freq, corpus))
    filtlen = len(corpus)
    LOGGER.end("filtered #%s (%s percent) words. New corpus size: %s (%s percent of original)" %
               (origlen - filtlen,
                float(origlen - filtlen) / float(origlen) * 100.0,
                filtlen,
                filtlen / float(origlen) * 100.0
               ))

    if nwords is not None:
        LOGGER.log("taking N(%s) words form the corpus: " % (nwords, ))
        corpus = corpus[:nwords]
    LOGGER.end()
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
    # vs * metric * ws^t = [S1 x EMBEDSIZE] x [EMBEDSIZE x S1] = [S1 x S2]

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
def normalize(vs, metric):
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


def mk_symmetric_mat(n):
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
    def __init__(self, embedsize):
        super(EuclidMetric, self).__init__()
        self.mat_ = nn.Parameter(torch.eye(embedsize).to(DEVICE), requires_grad=False)
    @property
    def mat(self):
        return self.mat_

class ReimannMetric(Metric):
    def __init__(self, embedsize):
        super(ReimannMetric, self).__init__()
        self.sqrt_ = nn.Parameter(torch.randn([embedsize, embedsize]).to(DEVICE), requires_grad=True)
    @property
    def mat(self):
        return torch.mm(self.sqrt_, self.sqrt_)

class PseudoReimannMetric(Metric):
    def __init__(self, embedsize):
        super(PseudoReimannMetric, self).__init__()
        self.mat_ = mk_symmetric_mat(embedsize)

    @property
    def mat(self):
        return self.mat_

class SkipGramOneHotDataset(Dataset):
    def __init__(self, LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE):
        self.TEXT = TEXT

        self.VOCAB = VOCAB
        self.VOCABSIZE = VOCABSIZE
        self.WINDOWSIZE = WINDOWSIZE

        LOGGER.start("creating I2W, W2I")
        self.I2W = dict(enumerate(VOCAB))
        self.W2I = { v: k for (k, v) in self.I2W.items() }
        LOGGER.end()

        LOGGER.start("counting frequency of words")
        self.W2F = mk_word_histogram(TEXT, VOCAB)
        LOGGER.end()

    def __getitem__(self, ix):
        focusix = ix // (2 * self.WINDOWSIZE)
        focusix += self.WINDOWSIZE
        deltaix = (ix % (2 * self.WINDOWSIZE)) - self.WINDOWSIZE

        return {'focusonehot': bow_vec([self.TEXT[focusix]], self.W2I, self.VOCABSIZE),
                'ctxtruelabel': self.W2I[self.TEXT[focusix + deltaix]]
                }

    def __len__(self):
        # we can't query the first or last value.
        # first because it has no left, last because it has no right
        return (len(self.TEXT) - 2 * self.WINDOWSIZE) * (2 * self.WINDOWSIZE)

class Word2ManSkipGramOneHot(nn.Module):
    def __init__(self, VOCABSIZE, EMBEDSIZE, DEVICE):
        super(Word2ManSkipGramOneHot, self).__init__()
        LOGGER.start("creating EMBEDM")
        self.EMBEDM = nn.Parameter(Variable(torch.randn(VOCABSIZE, EMBEDSIZE).to(DEVICE), requires_grad=True))
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
        xsembeds_dots_embeds = dots(xsembeds, self.EMBEDM, metric)
        # TODO: why is this correct? I don't geddit.
        # what in the fuck does it mean to log softmax cosine?
        # [BATCHSIZE x VOCABSIZE]
        xsembeds_dots_embeds = F.log_softmax(xsembeds_dots_embeds, dim=1)

        return xsembeds_dots_embeds

    def train(self, traindata, metric):
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
    def make_dataset(cls, LOGGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE):
        """
        create a dataset from the given text
        """
        return SkipGramOneHotDataset(LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE)


class CBOWDataset(Dataset):
    def __init__(self, LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE):
        self.TEXT = TEXT

        self.VOCAB = VOCAB
        self.VOCABSIZE = VOCABSIZE
        self.WINDOWSIZE = WINDOWSIZE

        LOGGER.start("creating I2W, W2I")
        self.I2W = dict(enumerate(VOCAB))
        self.W2I = { v: k for (k, v) in self.I2W.items() }
        LOGGER.end()

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
    def __init__(self, VOCABSIZE, EMBEDSIZE):
        super(Word2ManCBOW, self).__init__()
        LOGGER.start("creating EMBEDM")
        self.EMBEDM = nn.Parameter(Variable(torch.randn(VOCABSIZE, EMBEDSIZE).to(DEVICE), requires_grad=True))
        # TODO: I have so. many. questions
        # 1. Why linear?
        self.embed2vocab = nn.Parameter(Variable(torch.randn(EMBEDSIZE, VOCABSIZE).to(DEVICE), requires_grad=True))
        LOGGER.end()

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

    def train(self, traindata, metric):
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
    def make_dataset(cls, LOGGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE):
        """
        create a dataset from the given text
        """
        return CBOWDataset(LOGGER, TEXT, VOCAB, VOCABSIZE, WINDOWSIZE)


class Parameters:
    """God object containing everything the model has"""
    def __init__(self, LOGGER, DEVICE):
        """default values"""
        self.EPOCHS = 100
        self.BATCHSIZE = 512
        self.EMBEDSIZE = 300
        self.LEARNING_RATE = 0.001
        self.WINDOWSIZE = 2
        self.NWORDS = 10000

        self.create_time = current_time_str()


    def init_model(self, metrictype, traintype,
                          metric_state_dict=None,
                          word2man_state_dict=None,
                          optimizer_state_dict=None):
        TEXT = load_corpus(LOGGER, self.NWORDS)
        assert ((metric_state_dict is None and word2man_state_dict is None
                and optimizer_state_dict is None) or
                (metric_state_dict is not None and word2man_state_dict is not None
                 and optimizer_state_dict is not None))

        self.metrictype = metrictype
        self.traintype = traintype

        LOGGER.start("building vocabulary")
        VOCAB = set(TEXT)
        VOCABSIZE = len(VOCAB)
        LOGGER.end()
        LOGGER.start("creating metric")

        if metrictype == "euclid":
            self.METRIC = EuclidMetric(self.EMBEDSIZE)
        elif metrictype == "reimann":
            self.METRIC = ReimannMetric(self.EMBEDSIZE)
        else:
            assert(metrictype == "pseudoreimann")
            self.METRIC = PseudoReimannMetric(self.EMBEDSIZE)
        # check that metric is subclass of Metric
        assert(self.METRIC is not None)
        # TODO: check that metric is a subclass
        if metric_state_dict is not None:
            self.METRIC.load_state_dict(metric_state_dict, strict=True)
        LOGGER.end()

        LOGGER.start("creating word2man")
        if traintype == "skipgramonehot":
            self.WORD2MAN = Word2ManSkipGramOneHot(VOCABSIZE, self.EMBEDSIZE, DEVICE)
        elif traintype == "skipgramnhot":
            raise RuntimeError("unimplemented n-hot skipgram")
        # self.WORD2MAN = Word2ManSkipGramHot(VOCABSIZE, self.EMBEDSIZE, DEVICE, metrictype)
        else:
            assert(traintype == "cbow")
            self.WORD2MAN = Word2ManCBOW(VOCABSIZE, self.EMBEDSIZE)

        if word2man_state_dict is not None:
            self.WORD2MAN.load_state_dict(word2man_state_dict, strict=True)
        LOGGER.end()

        LOGGER.start("creating OPTIMISER")
        self.optimizer = optim.Adam(itertools.chain(self.WORD2MAN.parameters(), self.METRIC.parameters()), lr=self.LEARNING_RATE)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        LOGGER.end()

        LOGGER.start("creating dataset...")
        self.DATASET = self.WORD2MAN.make_dataset(LOGGER,
                                                  TEXT,
                                                  VOCAB,
                                                  VOCABSIZE,
                                                  self.WINDOWSIZE)
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
            "EPOCHS": self.EPOCHS,
            "BATCHSIZE": self.BATCHSIZE,
            "EMBEDSIZE": self.EMBEDSIZE,
            "LEARNINGRATE": self.LEARNING_RATE,
            "WINDOWSIZE": self.WINDOWSIZE,
            "NWORDS": self.NWORDS,
            "CREATE_TIME": self.create_time,
            "METRICTYPE": self.metrictype,
            "TRAINTYPE": self.traintype,
            "WORD2MAN": self.WORD2MAN.state_dict(),
            "METRIC": self.METRIC.state_dict(),
            "OPTIMIZER": self.optimizer.state_dict()
        }
        return st

    def load_model_state_dict(self, state):
        self.EPOCHS = state["EPOCHS"]
        self.BATCHSIZE = state["BATCHSIZE"]
        self.EMBEDSIZE = state["EMBEDSIZE"]
        self.LEARNING_RATE = state["LEARNINGRATE"]
        self.WINDOWSIZE = state["WINDOWSIZE"]
        self.NWORDS = state["NWORDS"]
        self.create_time = state["CREATE_TIME"]
        self.metrictype = state["METRICTYPE"]
        self.traintype = state["TRAINTYPE"]

        self.init_model(metrictype=self.metrictype,
                               traintype=self.traintype,
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


def word_to_embed_vector(w):
    """
    find the embedded vector of word w
    returns: [1 x EMBEDSIZE]
    """
    v = PARAMS.WORD2MAN.EMBEDM[PARAMS.DATASET.W2I[w], :]
    return v.view(1, -1)

def test_find_close_vectors(v):
    """ Find vectors close to w in the normalized embedding"""
    # dot [1 x EMBEDSIZE] [VOCABSIZE x EMBEDSIZE] = [1 x VOCABSIZE]
    EMBEDNORM = normalize(PARAMS.WORD2MAN.EMBEDM, PARAMS.METRIC.mat).to(DEVICE)
    v = normalize(v, PARAMS.METRIC.mat)
    wix2sim = dots(v, EMBEDNORM, PARAMS.METRIC.mat)

    wordweights = [(PARAMS.DATASET.I2W[i], wix2sim[0][i].item()) for i in range(PARAMS.DATASET.VOCABSIZE)]
    # The story of floats which cannot be ordered. I found
    # out I need this filter once I found this entry in the
    # table: ('classified', nan)
    wordweights = list(filter(lambda ww: not math.isnan(ww[1]), wordweights))
    # sort AFTER removing NaNs
    wordweights.sort(key=lambda wdot: wdot[1], reverse=True)

    return wordweights




def cli_prompt():
    """Call to launch prompt interface."""

    LOGGER.start("normalizing EMBEDM...")
    PARAMS.WORD2MAN.EMBEDM = PARAMS.WORD2MAN.EMBEDM.to(DEVICE)
    # PARAMS.METRIC.mat = PARAMS.METRIC.mat.to(DEVICE)
    LOGGER.end("done.")
    def prompt_word(session):
        """Prompt for a word and print the closest vectors to the word"""
        # [VOCABSIZE x EMBEDSIZE]
        COMPLETER = ThreadedCompleter(WordCompleter(PARAMS.DATASET.VOCAB))

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
            wordweights = test_find_close_vectors(word_to_embed_vector(raw[1]))
            for (word, weight) in wordweights[:15]:
                print_formatted_text("\t%s: %s" % (word, weight))
        elif raw[0] == "sim":
            if len(raw) != 4:
                print_formatted_text("error: expected sim <w1> <w2> <w3>")
                return
            v1 = word_to_embed_vector(raw[1])
            v2 = word_to_embed_vector(raw[2])
            v3 = word_to_embed_vector(raw[3])
            vsim = normalize(v1 - v2 + v3, PARAMS.METRIC.mat)
            wordweights = test_find_close_vectors(vsim)
            for (word, weight) in wordweights[:15]:
                print_formatted_text("\tnormal(a - b + c) %s: %s" % (word, weight))


            v1 = normalize(word_to_embed_vector(raw[1]), PARAMS.METRIC.mat)
            v2 = normalize(word_to_embed_vector(raw[2]), PARAMS.METRIC.mat)
            v3 = normalize(word_to_embed_vector(raw[3]), PARAMS.METRIC.mat)

            vsim = normalize(v2 - v1 + v3, PARAMS.METRIC.mat)
            wordweights = test_find_close_vectors(vsim)
            for (word, weight) in wordweights[:15]:
                print_formatted_text("\tnormal(normal(king) - normal(man) + normal(woman)): %s: %s" % (word, weight))

        elif raw[0] == "dot":
            if len(raw) != 3:
                print_formatted_text("error: expected dot <w1> <w2>")
                return

            v1 = normalize(word_to_embed_vector(raw[1]), PARAMS.METRIC.mat)
            v2 = normalize(word_to_embed_vector(raw[2]), PARAMS.METRIC.mat)
            print_formatted_text("\t%s" % (dots(v1, v2, PARAMS.METRIC.mat), ))
        elif raw[0] == "metric":
            print_formatted_text(PARAMS.METRIC.mat)
            w,v = torch.eig(PARAMS.METRIC.mat,eigenvectors=True)
            print_formatted_text("eigenvalues:\n%s" % (w, ))
            print_formatted_text("eigenvectors:\n%s" % (v, ))
            (s, v, d) = torch.svd(PARAMS.METRIC.mat)
            print_formatted_text("SVD :=\n%s\n%s\n%s" % (s, v, d))
        else:
            print_formatted_text("invalid command, type ? for help")

    session = PromptSession()
    while True:
        try:
            prompt_word(session)
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        except KeyError as e:
            print_formatted_text("exception:\n%s" % (e, ))


# =========== Actual code ============
LOGGER = TimeLogger()


# setup device
LOGGER.start("setting up device")
DEVICE = torch.device(1) if torch.cuda.is_available() else torch.device('cpu')
LOGGER.end("device: %s" % DEVICE)


PARAMS = Parameters(LOGGER, DEVICE)
if PARSED.loadpath is not None:
    LOGGER.start("loading model from: %s" % (PARSED.loadpath))
    # pass the device so that the tensors live on the correct device.
    # this might be stale since we were on a different device before.
    try:
        PARAMS = torch.load(PARSED.loadpath, map_location=DEVICE)
        #PARAMS.load_model_state_dict(torch.load(PARSED.loadpath, map_location=DEVICE))
        LOGGER.end("loaded params from: %s" % PARAMS.create_time)
    except FileNotFoundError:
        PARAMS.init_model(traintype=PARSED.traintype, metrictype=PARSED.metrictype)
        LOGGER.end("file (%s) not found. Creating new model" % (PARSED.loadpath, ))
else:
    LOGGER.start("no loadpath given. Creating fresh parameters...")
    PARAMS.init_model(traintype=PARSED.traintype, metrictype=PARSED.metrictype)
    LOGGER.end()

# @EXPERIMENT.capture
def traincli(savepath):
    def save():
        LOGGER.start("\nsaving model to: %s" % (savepath))
        with open(savepath, "wb") as sf:
            # torch.save(PARAMS.get_model_state_dict(), sf)
            torch.save(PARAMS, sf)
        LOGGER.end()


    # Notice that we do not normalize the vectors in the hidden layer
    # when we train them! this is intentional: In general, these vectors don't
    # seem to be normalized by most people, so it's weird if we begin to
    # normalize them.
    # Read also: what is the meaning of the length of a vector in word2vec?
    # https://stackoverflow.com/questions/36034454/what-meaning-does-the-length-of-a-word2vec-vector-have
    bar = progressbar.ProgressBar(max_value=math.ceil(PARAMS.EPOCHS * len(PARAMS.DATALOADER)))
    loss_sum = 0
    ix = 0
    time_last_save = datetime.datetime.now()
    time_last_print = datetime.datetime.now()
    last_print_ix = 0
    for epoch in range(PARAMS.EPOCHS):
        for traindata in PARAMS.DATALOADER:
            ix += 1
            PARAMS.optimizer.zero_grad()   # zero the gradient buffers
            l = PARAMS.WORD2MAN.train(traindata, PARAMS.METRIC.mat)
            loss_sum += l.item()
            l.backward()
            PARAMS.optimizer.step()
            bar.update(bar.value + 1)
            # updating data
            now = datetime.datetime.now()

            # printing
            TARGET_PRINT_TIME_IN_S = 1
            if (now - time_last_print).seconds >= TARGET_PRINT_TIME_IN_S:
                nbatches = ix - last_print_ix
                print("\nLOSSES sum: %s | avg per batch(#batch=%s): %s | avg per elements(#elems=%s): %s" %
                      (loss_sum,
                       nbatches,
                       loss_sum / nbatches,
                       nbatches * PARAMS.BATCHSIZE,
                       loss_sum / (nbatches * PARAMS.BATCHSIZE)))
                loss_sum = 0
                time_last_print = now
                last_print_ix = ix

            # saving
            TARGET_SAVE_TIME_IN_S = 15 * 60 # save every 15 minutes
            if (now - time_last_save).seconds > TARGET_SAVE_TIME_IN_S:
                save()
                time_last_save = now
    save()

from nltk.corpus import wordnet
from random import sample
def evaluate():
    # PARAMS.DATASET.VOCAB
    # PARAMS.DATASET.TEXT
    # word_to_embed_vector
    # test_find_close_vectors
    # dots -> dot products
    # cosine_sim -> cosine similarity

    # Get a sample of unique words (VOCAB)
    # For each word in the vocab, find the 10 closest vectors
    # Find these words
    # Find wup_similarity and path_similarity of the words in the list

    word_vector_pairs = []
    word_sym_pairs = []

    sampled_words = sample(PARAMS.DATASET.VOCAB, 10)
    for word in sampled_words:
        similar_words = test_find_close_vectors(word_to_embed_vector(word))[:10]
        word_sym_pairs.append((word, similar_words))

    # Approximation made here for now: Only the first sense of each similar word has been taken
    # Later task: Will compare similarities from the entire synset.

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

                        rows.append([word, sim_word, max(wup), max(path), score])
                        # print(word + "\t" + sim_word + "\t" + str(wup_sim) + "\t" + str(path_sim) + "\t" + str(score))
                    except IndexError as e:
                        print("%word not in wordnet: %s" % sim_word)
            print(tabulate.tabulate(rows, headers=["Word", "Similar", "WUP Score", "Path Score", "Our Product"]))

            # What are the closest words according to wordnet that belong in our corpus?

#            wn_max_corpus = []
#            for corpus_word in PARAMS.DATASET.VOCABS:
#                wn_corpus_word = wordnet.synsets(corpus_word)
#                cor_wup = [wordnet.wup_similarity(sense, cor_sense) for sense in wn_word for cor_sense in wn_corpus_word]
#                cor_wup = [0.0 if w==None else w for w in wup]
#                wn_max_corpus.append((wn_corpus_word, max(cor_wup)))


# @EXPERIMENT.main
def main():
    if PARSED.command == "train":
        traincli(PARSED.savepath)
    elif PARSED.command == "test":
        cli_prompt()
    elif PARSED.command == "eval":
        evaluate()
    else:
        raise RuntimeError("unknown command: %s" % PARSED.command)

if __name__ == "__main__":
    main()

