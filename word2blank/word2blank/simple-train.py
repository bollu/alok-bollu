#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
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
import signal
import sys
import click
import datetime 
import math
import progressbar
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit import prompt
from prompt_toolkit import print_formatted_text


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
    def __init__(self):
        pass
    def start(self, toprint):
        self.t = datetime.datetime.now()
        print(str(toprint) + "...", end="")
        sys.stdout.flush()

    def end(self, toprint=None):
        now = datetime.datetime.now()
        if(toprint): print(toprint)
        print("Done. time: %s" % (now - self.t))
        if (toprint): print("--")
        sys.stdout.flush()

def load_corpus(LOGGER):

    def flatten(ls):
        return [item for sublist in ls for item in sublist]

    # return  """we are about to study the idea of a computational process.
    #  computational processes are abstract beings that inhabit computers.
    #  as they evolve, processes manipulate other abstract things called data.
    #  the evolution of a process is directed by a pattern of rules called a program.
    #  people create programs to direct processes.
    #  in effect, we conjure the spirits of the computer with our spells.""".split()                                                                
    CORPUS_NAME = "text8"
    LOGGER.start("loading corpus: %s" % CORPUS_NAME)
    try:
        sys.path.insert(0, api.base_dir)
        module = __import__(CORPUS_NAME)
        corpus = module.load_data()
    except Exception as e:
        print("unable to find text8 locally.\nERROR: %s" % (e, ))
        print("Downloading using gensim-data...")
        corpus = api.load(CORPUS_NAME)
        print("Done.")

    corpus = list(corpus)
    print("number of documents in corpus: %s" % (len(corpus), ))
    DOCS_TO_TAKE = 1
    print("taking first N(%s) documents in corpus: %s" % (DOCS_TO_TAKE, DOCS_TO_TAKE))
    corpus = corpus[:DOCS_TO_TAKE]
    corpus = flatten(corpus)
    print("number of words in corpus: %s" % (len(corpus), ))
    LOGGER.end()
    return corpus


def batch(xs, BATCHSIZE):
    ix = 0
    while ix + BATCHSIZE < len(xs):
        data = xs[ix:ix+BATCHSIZE]
        ix += BATCHSIZE
        yield data


def norm(v, w, metric):
    dot = torch.mm(torch.mm(v.view(1, -1), metric), w.view(-1, 1))
    return dot


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
    
def normalize(vs, metric):
    # vs = [S1 x EMBEDSIZE]
    # out = [S1 x EMBEDSIZE]

    # vs_dots_vs = [S1 x S1]
    vs_dots_vs = dots(vs, vs, metric)
    # norm = S1 x 1
    norm = torch.sqrt(torch.diag(vs_dots_vs)).view(-1, 1)
    # normvs = S1 x 1
    normvs =  vs / norm

    ERROR_THRESHOLD = 0.1
    assert(torch.norm(torch.diag(dots(normvs, normvs, metric)) - torch.ones(len(vs)).to(DEVICE)).item() < ERROR_THRESHOLD)
    return normvs

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


class Parameters:
    def __init__(self, LOGGER, DEVICE, VOCABSIZE):
        self.EPOCHS = 5
        self.BATCHSIZE = 4
        self.EMBEDSIZE = 100
        self.LEARNING_RATE = 0.1
        self.WINDOWSIZE = 2
        LOGGER.start("creating EMBEDM")
        self.EMBEDM = nn.Parameter(Variable(torch.randn(VOCABSIZE, self.EMBEDSIZE).to(DEVICE), requires_grad=True))
        LOGGER.end()

        # metric, currently identity matrix
        LOGGER.start("creating metric")
        self.METRIC = torch.eye(self.EMBEDSIZE).to(DEVICE)
        LOGGER.end()
# =========== Actual code ============
LOGGER = TimeLogger()

# setup device
LOGGER.start("setting up device")
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')
LOGGER.end("device: %s" % DEVICE)


TEXT = load_corpus(LOGGER)
LOGGER.start("filtering stopwords")
TEXT = list(filter(lambda w: w not in STOPWORDS, TEXT))
LOGGER.end()

VOCAB = set(TEXT)
VOCABSIZE = len(VOCAB)

LOGGER.start("creating i2w, w2i")
i2w = dict(enumerate(VOCAB))
w2i = { v: k for (k, v) in i2w.items() }
LOGGER.end()


params = Parameters(LOGGER, DEVICE, VOCABSIZE)

def hot(ws, w2i):
    """
    hot vector for each word in ws
    """
    v = Variable(torch.zeros(VOCABSIZE).float())
    for w in ws: v[w2i[w]] = 1.0
    return v

def test_find_close_vectors(w, normalized_embed):
    """ Find vectors close to w """
    # [1 x VOCABSIZE] 
    whot = hot([w], w2i).to(DEVICE)
    # [1 x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [1 x EMBEDSIZE]
    wembed = normalize(torch.mm(whot.view(1, -1), params.EMBEDM), params.METRIC)

    # dot [1 x EMBEDSIZE] [VOCABSIZE x EMBEDSIZE] = [1 x VOCABSIZE]
    wix2sim = dots(wembed, normalized_embed, params.METRIC)

    wordweights = [(i2w[i], wix2sim[0][i].item()) for i in range(VOCABSIZE)]
    wordweights.sort(key=lambda wdot: wdot[1], reverse=True)

    return wordweights


def test():
    # [VOCABSIZE x EMBEDSIZE]
    EMBEDNORM = normalize(params.EMBEDM, params.METRIC)
    COMPLETER = WordCompleter(VOCAB)

    ws = prompt("type in word>", completer=COMPLETER).split()
    for w in ws:
        whot = hot([w], w2i).to(DEVICE)
        # [1 x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [1 x EMBEDSIZE]
        wembed = normalize(torch.mm(whot.view(1, -1), params.EMBEDM), params.METRIC)

        wordweights = test_find_close_vectors(w, EMBEDNORM)
        for (word, weight) in wordweights[:10]:
            print("\t%s: %s" % (word, weight))

@click.group()
def cli():
    pass

@click.command()
@click.option('--loadpath', default=None, help='Path to load model from')
@click.option('--savepath',
              default=("save-auto-%s" % (datetime.datetime.now(), )),
                       help='Path to save model from')
def traincli(loadpath, savepath):
    global params
    def save():
        LOGGER.start("saving model to %s" % (savepath))
        with open(savepath, "wb") as sf:
            torch.save(params, sf)
            LOGGER.end()

    if loadpath is not None:
        params = torch.load(loadpath)

    LOGGER.start("creating DATA\n")
    DATA = []
    for i in progressbar.progressbar(range((len(TEXT) - 2 * params.WINDOWSIZE))):
        ix = i + params.WINDOWSIZE

        wsfocus = [TEXT[ix]]
        wsctx = [TEXT[ix + deltaix] for deltaix in range(-params.WINDOWSIZE, params.WINDOWSIZE + 1)]
        DATA.append(torch.stack([hot(wsfocus, w2i), hot(wsctx, w2i)]))
    DATA = torch.stack(DATA)
    LOGGER.end()

    LOGGER.start("creating optimizer and loss function")
    optimizer = optim.SGD([params.EMBEDM], lr=params.LEARNING_RATE)
    loss = nn.MSELoss()
    LOGGER.end()


    # Notice that we do not normalize the vectors in the hidden layer
    # when we train them! this is intentional: In general, these vectors don't
    # seem to be normalized by most people, so it's weird if we begin to
    # normalize them. 
    # Read also: what is the meaning of the length of a vector in word2vec?
    # https://stackoverflow.com/questions/36034454/what-meaning-does-the-length-of-a-word2vec-vector-have

    with progressbar.ProgressBar(max_value=params.EPOCHS * len(DATA) / params.BATCHSIZE) as bar:
        loss_sum = 0
        ix = 0
        for epoch in range(params.EPOCHS):
            for train in batch(DATA, params.BATCHSIZE):
                ix += 1
                # [BATCHSIZE x VOCABSIZE]
                xs = train[:, 0].to(DEVICE)
                # [BATCHSIZE x VOCABSIZE]
                ysopt = train[:, 1].to(DEVICE)

                optimizer.zero_grad()   # zero the gradient buffers
                # embedded vectors of the batch vectors
                # [BATCHSIZE x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [BATCHSIZE x EMBEDSIZE]
                xsembeds = torch.mm(xs, params.EMBEDM)

                # dots(BATCHSIZE x EMBEDSIZE], 
                #     [VOCABSIZE x EMBEDSIZE],
                #     [EMBEDSIZE x EMBEDSIZE]) = [BATCHSIZE x VOCABSIZE]
                xs_dots_embeds = dots(xsembeds, params.EMBEDM, params.METRIC)

                l = loss(ysopt, xs_dots_embeds)
                loss_sum += torch.sum(torch.abs(l)).item()
                l.backward()
                optimizer.step()
                bar.update(bar.value + 1)

                PRINT_PER_NUM_ELEMENTS = 10000
                PRINT_PER_NUM_BATCHES = PRINT_PER_NUM_ELEMENTS // params.BATCHSIZE
                if (ix % PRINT_PER_NUM_BATCHES == 0):
                    print("LOSSES sum: %s | avg per batch: %s | avg per elements: %s" % 
                          (loss_sum,
                           loss_sum / PRINT_PER_NUM_BATCHES,
                           loss_sum / PRINT_PER_NUM_ELEMENTS))
                    loss_sum = 0

                SAVE_PER_NUM_ELEMENTS = 20000
                SAVE_PER_NUM_BATCHES = PRINT_PER_NUM_ELEMENTS // params.BATCHSIZE

                if ix % SAVE_PER_NUM_BATCHES == SAVE_PER_NUM_BATCHES - 1:
                    save()
    
    save()


@click.command()
@click.argument("loadpath")
def testcli(loadpath):
    global params

    with open(loadpath, "rb") as lf:
        params = torch.load(lf)
    test()

cli.add_command(traincli)
cli.add_command(testcli)
if __name__ == "__main__":
    cli()