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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
    """Provide an API to log start and end times of tasks"""
    def __init__(self):
        self.ts = []
        pass
    def start(self, toprint):
        depth = len(self.ts)
        self.ts.append(datetime.datetime.now())
        print(" " * 4 * depth + str(toprint) + "...")
        sys.stdout.flush()

    def end(self, toprint=None):
        depth = len(self.ts)

        start = self.ts.pop()
        now = datetime.datetime.now()

        if(toprint): print(toprint)
        print(" " * 4 * depth + "Done. time: %s" % (now - start))
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


def current_time_str():
    """Return the current time as a string"""
    return datetime.datetime.now().strftime("%X-%a-%b")


class Parameters:
    """God object containing everything the model has"""
    def __init__(self, LOGGER, DEVICE, TEXT, VOCAB, VOCABSIZE):
        """default values"""
        self.EPOCHS = 5
        self.BATCHSIZE = 4
        self.EMBEDSIZE = 100
        self.LEARNING_RATE = 0.1
        self.WINDOWSIZE = 2
        self.create_time = current_time_str()

        LOGGER.start("creating EMBEDM")
        self.EMBEDM = nn.Parameter(Variable(torch.randn(VOCABSIZE, self.EMBEDSIZE).to(DEVICE), requires_grad=True))
        LOGGER.end()

        # metric, currently identity matrix
        LOGGER.start("creating metric")
        self.METRIC = torch.eye(self.EMBEDSIZE).to(DEVICE)
        LOGGER.end()

        LOGGER.start("creating OPTIMISER")
        self.optimizer = optim.SGD([self.EMBEDM], lr=self.LEARNING_RATE)
        LOGGER.end()


        LOGGER.start("creating dataset...")
        self.DATASET = SkipGramDataset(LOGGER, TEXT, VOCAB, VOCABSIZE, self.WINDOWSIZE)
        LOGGER.end()

        LOGGER.start("creating DATA\n")
        self.DATA = DataLoader(self.DATASET, batch_size=self.BATCHSIZE, shuffle=True)
        LOGGER.end()

def mk_word_histogram(ws, vocab):
    """count frequency of words in words, given vocabulary size."""
    w2f = { w : 0 for w in vocab }
    for w in ws:
        w2f[w] += 1
    return w2f

class SkipGramDataset(Dataset):
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

    def __getitem__(self, idx):
        idx = idx + self.WINDOWSIZE
        wsfocusix = [self.TEXT[idx]]
        wsctxix = [self.TEXT[idx + deltaix] for deltaix in range(-self.WINDOWSIZE, self.WINDOWSIZE + 1)]

        return torch.stack([hot(wsfocusix, self.W2I, self.VOCABSIZE), hot(wsctxix, self.W2I, self.VOCABSIZE)])

    def __len__(self):
        # we can't query the first or last value.
        # first because it has no left, last because it has no right
        return (len(self.TEXT) - 2 * self.WINDOWSIZE) 


def hot(ws, W2I, VOCABSIZE):
    """
    hot vector for each word in ws
    """
    v = Variable(torch.zeros(VOCABSIZE).float())
    for w in ws: v[W2I[w]] = 1.0
    return v


def cli_prompt():
    """Call to launch prompt interface."""

    def test_find_close_vectors(w, normalized_embed):
        """ Find vectors close to w in the normalized embedding"""
        # [1 x VOCABSIZE] 
        whot = hot([w], PARAMS.W2I, PARAMS.VOCABSIZE).to(DEVICE)
        # [1 x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [1 x EMBEDSIZE]
        wembed = normalize(torch.mm(whot.view(1, -1), PARAMS.EMBEDM), PARAMS.METRIC)

        # dot [1 x EMBEDSIZE] [VOCABSIZE x EMBEDSIZE] = [1 x VOCABSIZE]
        wix2sim = dots(wembed, normalized_embed, PARAMS.METRIC)

        wordweights = [(PARAMS.I2W[i], wix2sim[0][i].item()) for i in range(VOCABSIZE)]
        wordweights.sort(key=lambda wdot: wdot[1], reverse=True)

        return wordweights


    def prompt_word():
        """Prompt for a word and print the closest vectors to the word"""
        # [VOCABSIZE x EMBEDSIZE]
        EMBEDNORM = normalize(PARAMS.EMBEDM, PARAMS.METRIC)
        COMPLETER = WordCompleter(VOCAB)

        ws = prompt("type in word>", completer=COMPLETER).split()
        for w in ws:
            wordweights = test_find_close_vectors(w, EMBEDNORM)
            for (word, weight) in wordweights[:10]:
                print("\t%s: %s" % (word, weight))
    while True:
        try:
            prompt_word()
        except Exception as e:
            print("exception:\n%s" % (e, ))

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


LOGGER.start("building vocabulary")
VOCAB = set(TEXT)
VOCABSIZE = len(VOCAB)
LOGGER.end()

LOGGER.start("creating parameters...")
PARAMS = Parameters(LOGGER, DEVICE, TEXT, VOCAB, VOCABSIZE)
LOGGER.end()



@click.group()
def cli():
    pass


def DEFAULT_MODELPATH():
    now = datetime.datetime.now()
    return "save-auto-%s.model" % (current_time_str(), )

@click.command()
@click.option('--loadpath', default=None, help='Path to load model')
@click.option('--savepath', default=DEFAULT_MODELPATH(), help='Path to save model')
def traincli(loadpath, savepath):
    global PARAMS
    def save():
        LOGGER.start("saving model to: %s" % (savepath))
        with open(savepath, "wb") as sf:
            torch.save(PARAMS, sf)
            LOGGER.end()

    if loadpath is not None:
        LOGGER.start("loading model from: %s" % (loadpath))
        PARAMS = torch.load(loadpath)
        LOGGER.end("loaded params from: %s" % PARAMS.create_time)


    LOGGER.start("creating optimizer and loss function")
    loss = nn.MSELoss()
    LOGGER.end()


    # Notice that we do not normalize the vectors in the hidden layer
    # when we train them! this is intentional: In general, these vectors don't
    # seem to be normalized by most people, so it's weird if we begin to
    # normalize them. 
    # Read also: what is the meaning of the length of a vector in word2vec?
    # https://stackoverflow.com/questions/36034454/what-meaning-does-the-length-of-a-word2vec-vector-have

    with progressbar.ProgressBar(max_value=math.ceil(PARAMS.EPOCHS * len(PARAMS.DATA))) as bar:
        loss_sum = 0
        ix = 0
        for epoch in range(PARAMS.EPOCHS):
            for train in PARAMS.DATA:
                ix += 1
                # [BATCHSIZE x VOCABSIZE]
                xs = train[:, 0].to(DEVICE)
                # [BATCHSIZE x VOCABSIZE]
                ysopt = train[:, 1].to(DEVICE)

                PARAMS.optimizer.zero_grad()   # zero the gradient buffers
                # embedded vectors of the batch vectors
                # [BATCHSIZE x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [BATCHSIZE x EMBEDSIZE]
                xsembeds = torch.mm(xs, PARAMS.EMBEDM)

                # dots(BATCHSIZE x EMBEDSIZE], 
                #     [VOCABSIZE x EMBEDSIZE],
                #     [EMBEDSIZE x EMBEDSIZE]) = [BATCHSIZE x VOCABSIZE]
                xs_dots_embeds = dots(xsembeds, PARAMS.EMBEDM, PARAMS.METRIC)

                l = loss(ysopt, xs_dots_embeds)
                loss_sum += torch.sum(torch.abs(l)).item()
                l.backward()
                PARAMS.optimizer.step()
                bar.update(bar.value + 1)

                PRINT_PER_NUM_ELEMENTS = 10000
                PRINT_PER_NUM_BATCHES = PRINT_PER_NUM_ELEMENTS // PARAMS.BATCHSIZE
                if (ix % PRINT_PER_NUM_BATCHES == 0):
                    print("LOSSES sum: %s | avg per batch: %s | avg per elements: %s" % 
                          (loss_sum,
                           loss_sum / PRINT_PER_NUM_BATCHES,
                           loss_sum / PRINT_PER_NUM_ELEMENTS))
                    loss_sum = 0

                SAVE_PER_NUM_ELEMENTS = 20000
                SAVE_PER_NUM_BATCHES = PRINT_PER_NUM_ELEMENTS // PARAMS.BATCHSIZE

                if ix % SAVE_PER_NUM_BATCHES == SAVE_PER_NUM_BATCHES - 1:
                    save()
        print("FINAL BAR VALUE: %s" % bar.value) 
    save()
    cli_prompt()


@click.command()
@click.argument("loadpath")
def testcli(loadpath):
    global PARAMS

    with open(loadpath, "rb") as lf:
        global PARAMS
        print("params: %s" % PARAMS)
        PARAMS = torch.load(lf)
        print("params: %s" % PARAMS)

    cli_prompt()


cli.add_command(traincli)
cli.add_command(testcli)
if __name__ == "__main__":
    cli()
