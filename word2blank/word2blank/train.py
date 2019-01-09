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
import signal
import sys
import click
import datetime 
 

LOSS_PRINT_STEP = 2000
def DEFAULT_MODELPATH():
    now = datetime.datetime.now()
    return now.strftime("%X-%a-%b") + ".model"

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
        yield ([s[i], sampler.sample(), 0])
        # yield ([s[i], sampler.sample(), 0])

def mk_onehot(sampler, w, device):
    # v = torch.zeros(len(sampler)).float()
    #v[sampler.wordix(w)] = 1.0
    # return v

    # TODO: understand why this does not work
    # return Variable(torch.LongTensor([sampler.wordix(w)], device=device))
    return Variable(torch.LongTensor([sampler.wordix(w)])).to(device)


# Word2Vec word2vec
# https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.ipynb
class Word2Vec(nn.Module):
    def __init__(self, sampler, nhidden):
        self.nhidden = nhidden
        nwords = len(sampler)
        """nwords: number of words"""
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(len(sampler), nhidden)

    # run the forward pass which matches word y in the context of x
    def forward(self, x_, y_):
        xembed = self.embedding(x_)
        # print("xembed: %s" % (xembed, ))
        xembed = xembed.view((self.nhidden,))
        # print("xembed: %s" % (xembed, ))

        yembed = self.embedding(y_)
        # print("yembed: %s" % (yembed, ))
        yembed = yembed.view((1, -1))
        yembed = yembed.view((self.nhidden,))

        score = torch.dot(xembed, yembed)
        log_probs = F.logsigmoid(score)
        # print("log_probs: %s" % (log_probs, ))
        return log_probs

# Corpus contains a list of sentences. Each s is a list of words
# Data pulled from:
# https://github.com/RaRe-Technologies/gensim-data
print ("loading corpus text8...")
corpus = api.load('text8') 
NSENTENCES = 10
corpus = list(itertools.islice(corpus, NSENTENCES))
print ("Done.")

# Count word frequencies
print("counting word frequencies...")
wfs = Counter()
for s in corpus:
    update_wfs(s, wfs)
sampler = Sampler(wfs)
print ("Done.")

@click.group()
def cli():
    pass

def torch_status_dump():
    print("---")
    print("CUDA available?: %s" % torch.cuda.is_available())
    print("GPU device: |%s|" % torch.cuda.get_device_name(0))
    print("---")

@click.command()
@click.option('--savepath', default=DEFAULT_MODELPATH(), help='Path to save model')
@click.option('--loadpath', default=None, help='Path to load model from')
def train(savepath, loadpath):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("device: %s" % device)

    # TODO: also save optimizer data so we can restart
    if loadpath is not None:
        print("loading model from %s" % loadpath)
        model = torch.load(loadpath)
        model.eval()
    else:
        model = Word2Vec(sampler, nhidden=100)

    assert (model)
    print("network: ")
    print(model)
    model.to(device)

    def signal_term_handler(signal, frame):
        print ("saving model to %s" % savepath)
        torch.save(model, savepath)
        sys.exit(0)
    signal.signal(signal.SIGTERM, signal_term_handler)
    signal.signal(signal.SIGINT, signal_term_handler)

    # optimise
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    last_print_time = datetime.datetime.now()
    running_loss = 0
    i = 0
    for epoch in range(2):
        for  s in corpus:
            for train in mk_skipgrams_sentence(s, sampler):
                i += 1
                # print("training on sample: %s" % (train,))
                (w, wctx, is_positive) = train
                #TODO: learn if this is actually the correct way of doing things...
                x_ = mk_onehot(sampler, w, device)
                y_ = mk_onehot(sampler, wctx, device)

                optimizer.zero_grad()   # zero the gradient buffers
                y = model(x_, y_)
                # print("y: %s" % y)
                loss = criterion(y, Variable(torch.Tensor([is_positive])).to(device))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                cur_print_time = datetime.datetime.now()
                if i % LOSS_PRINT_STEP == LOSS_PRINT_STEP - 1:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f | time: %s' %
                          (epoch + 1, i + 1, running_loss / LOSS_PRINT_STEP, 
                           (cur_print_time - last_print_time)))
                    last_print_time = cur_print_time
                    running_loss = 0.0


@click.command()
@click.argument('modelpath')
def test(modelpath):
    model = torch.load(modelpath)
    model.eval()
    criterion = nn.MSELoss()

    # list of all words
    WORDS = list(set(itertools.chain.from_iterable(corpus)))

    # find closest vectors to each word in the model
    with torch.no_grad():
        for wcur in WORDS[:50]:
            ws = []
            print ("curword: %s " % (wcur, ))
            for wother in WORDS:
                vcur = mk_onehot(sampler, wcur)
                vother = mk_onehot(sampler, wother)

                d = model(vcur, vother)
                ws.append((wother, d))
            ws.sort(key=lambda wd: wd[1], reverse=True)
            ws = ws [:10]
            print ("\n".join(["\t%s -> %s" % (w, d) for (w, d) in ws]))

cli.add_command(train)
cli.add_command(test)

if __name__ == "__main__":
    torch_status_dump()
    cli()

