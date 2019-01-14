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
    p.add_argument("--enforce-clean", action='store_true', help="enfore repo to be clean")

    sub = p.add_subparsers(dest="command")
    train = sub.add_parser("train", help="train the model")
    train.add_argument("--loadpath", help="path to model file to load from", default=None)
    train.add_argument("--savepath", help="path to save model to", default=DEFAULT_MODELPATH())

    test = sub.add_parser("test", help="test the model")
    test.add_argument("loadpath",  help="path to model file")
    
    return p.parse_args(s)
# if launching from the shell, parse first, then start loading datasets...
if __name__ == "__main__":
    global PARSED
    PARSED = parse(sys.argv[1:])
    assert (PARSED.command in ["train", "test"])

import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import gensim.downloader as api
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import math
import prompt_toolkit
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit import print_formatted_text
import sacred
import sacred.observers
import progressbar


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

def load_corpus(LOGGER, nwords):
    """load the corpus, and pull nwords from the corpus"""
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
    corpus = flatten(corpus)
    print("number of words in corpus (original): %s" % (len(corpus), ))

    LOGGER.start("filtering stopwords")
    corpus = list(filter(lambda w: w not in STOPWORDS, corpus))
    LOGGER.end()
    print("number of words in corpus (after filtering: %s" % (len(corpus), ))
    print("taking N(%s) words form the corpus: " % (nwords, ))
    corpus = corpus[:nwords]
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
    """God object containing everything the model has"""
    def __init__(self, LOGGER, DEVICE):
        """default values"""
        self.EPOCHS = 3
        self.BATCHSIZE = 128
        self.EMBEDSIZE = 200
        self.LEARNING_RATE = 0.025
        self.WINDOWSIZE = 2
        self.NWORDS = self.BATCHSIZE * 10000
        self.create_time = current_time_str()

        TEXT = load_corpus(LOGGER, self.NWORDS)


        LOGGER.start("building vocabulary")
        VOCAB = set(TEXT)
        VOCABSIZE = len(VOCAB)
        LOGGER.end()


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

        # TODO: pytorch dataloader is sad since it doesn't save state.
        # make a version that does save state.
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
        whot = hot([w], PARAMS.DATASET.W2I, PARAMS.DATASET.VOCABSIZE).to(DEVICE)
        # [1 x VOCABSIZE] x [VOCABSIZE x EMBEDSIZE] = [1 x EMBEDSIZE]
        wembed = normalize(torch.mm(whot.view(1, -1), PARAMS.EMBEDM), PARAMS.METRIC)

        # dot [1 x EMBEDSIZE] [VOCABSIZE x EMBEDSIZE] = [1 x VOCABSIZE]
        wix2sim = dots(wembed, normalized_embed, PARAMS.METRIC)

        wordweights = [(PARAMS.DATASET.I2W[i], wix2sim[0][i].item()) for i in range(PARAMS.DATASET.VOCABSIZE)]
        wordweights.sort(key=lambda wdot: wdot[1], reverse=True)

        return wordweights


    def prompt_word(session):
        """Prompt for a word and print the closest vectors to the word"""
        # [VOCABSIZE x EMBEDSIZE]
        EMBEDNORM = normalize(PARAMS.EMBEDM, PARAMS.METRIC)
        COMPLETER = WordCompleter(PARAMS.DATASET.VOCAB)

        ws = session.prompt("type in word>", completer=COMPLETER).split()
        for w in ws:
            wordweights = test_find_close_vectors(w, EMBEDNORM)
            for (word, weight) in wordweights[:10]:
                print_formatted_text("\t%s: %s" % (word, weight))

    session = PromptSession()
    while True:
        try:
            prompt_word(session)
        except Exception as e:
            print_formatted_text("exception:\n%s" % (e, ))


# =========== Actual code ============
LOGGER = TimeLogger()


# setup device
LOGGER.start("setting up device")
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')
LOGGER.end("device: %s" % DEVICE)


PARAMS = Parameters(LOGGER, DEVICE)

EXPERIMENT = sacred.Experiment()
EXPERIMENT.add_config(EPOCHS = PARAMS.EPOCHS,
                      BATCHSIZE = PARAMS.BATCHSIZE,
                      EMBEDSIZE = PARAMS.EMBEDSIZE,
                      LEARNING_RATE = PARAMS.LEARNING_RATE,
                      WINDOWSIZE = PARAMS.WINDOWSIZE,
                      NWORDS = PARAMS.NWORDS)
EXPERIMENT.observers.append(sacred.observers.FileStorageObserver.create('runs'))


@EXPERIMENT.capture
def traincli(loadpath, savepath):
    global PARAMS
    def save():
        LOGGER.start("\nsaving model to: %s" % (savepath))
        with open(savepath, "wb") as sf:
            torch.save(PARAMS, sf)
        EXPERIMENT.add_artifact(savepath)
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

    bar =  progressbar.ProgressBar(max_value=math.ceil(PARAMS.EPOCHS * len(PARAMS.DATA)))
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
                print("\nLOSSES sum: %s | avg per batch: %s | avg per elements: %s" % 
                      (loss_sum,
                       loss_sum / PRINT_PER_NUM_BATCHES,
                       loss_sum / PRINT_PER_NUM_ELEMENTS))
                loss_sum = 0

            SAVE_PER_NUM_ELEMENTS = 20000
            SAVE_PER_NUM_BATCHES = PRINT_PER_NUM_ELEMENTS // PARAMS.BATCHSIZE

            if ix % SAVE_PER_NUM_BATCHES == SAVE_PER_NUM_BATCHES - 1:
                save()
    save()


def testcli(loadpath):
    global PARAMS

    with open(loadpath, "rb") as lf:
        global PARAMS
        PARAMS = torch.load(lf)

    cli_prompt()



@EXPERIMENT.main
def main():
    if PARSED.command == "train":
        traincli(PARSED.loadpath, PARSED.savepath)
    elif PARSED.command == "test":
        testcli(PARSED.loadpath)
    else:
        raise RuntimeError("unknown command: %s" % PARSED.command)

if __name__ == "__main__":
    EXPERIMENT.run()

