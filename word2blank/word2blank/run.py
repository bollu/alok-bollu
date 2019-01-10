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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import signal
import sys
import click
import datetime 
import os
 

LEARNING_RATE=0.05
LOSS_PRINT_STEP = 10
BATCH_SIZE = 50
EPOCHS = 2000
NHIDDEN = 300

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


def mk_onehot(sampler, w):
    # v = torch.zeros(len(sampler)).float()
    #v[sampler.wordix(w)] = 1.0
    # return v

    # TODO: understand why this does not work
    # return Variable(torch.LongTensor([sampler.wordix(w)], device=device))
    return Variable(torch.LongTensor([sampler.wordix(w)]))

class SentenceSkipgramDataset(Dataset):
    def __init__(self, s, sampler):
        self.s = s
        self.sampler = sampler
        self.NNEGATIVES = len(self.s)
        self.NPOSITIVES = (len(self.s) - 2)

    def __getitem__(self, idx):
        # move index by 1 so we are in range [1..len - 2]
        # TODO: generalize this to a window.
        if (idx >= self.NPOSITIVES):
             idx -= self.NPOSITIVES
             x_ = self.s[idx]
             y_ = sampler.sample()
             is_positive_ = 0
        else:
            idx += 1
            assert(idx >= 1)
            assert(idx <= len(self.sampler) - 2)

            if idx % 2 == 0:
                (x_, y_, is_positive_) = (self.s[idx], self.s[idx - 1], 1)
            else:
                (x_, y_, is_positive_) = (self.s[idx], self.s[idx + 1], 1)
        x_ = mk_onehot(sampler, x_)
        y_ = mk_onehot(sampler, y_)
        is_positive_ = Variable(torch.LongTensor([is_positive_]))
        out =  torch.cat([x_, y_, is_positive_])
        return out

    def __len__(self):
        # we can't query the first or last value.
        # first because it has no left, last because it has no right
        return self.NPOSITIVES + self.NNEGATIVES



def mk_skipgrams_sentence_dataset(s, sampler, device):
    """s: current sentence, sampler: sampler, device: current device
       returns: torch.DataSet of values
    """
    # TODO: bench this. It is quite likely that creating all the tensors
    # on the CPU and then sending them to the GPU in one shot is way better
    # than piecemeal allocating on the GPU(?)
    out = []
    for (w, wctx, is_positive) in list(mk_skipgrams_sentence(s, sampler))[:10]:
        x_ = mk_onehot(sampler, w, device)
        y_ = mk_onehot(sampler, wctx, device)
        is_positive_ = Variable(torch.LongTensor([is_positive])).to(device)
        vec_ = torch.cat([x_, y_, is_positive_])
        out.append(vec_)
    print ("out: %s" % out)
    out = torch.stack(out)
    print ("out: %s" % out)
    out = TensorDataset(out)
    return out

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
        # 2d vectors of data x batch_size
        # TODO: is batch_size outer or inner dim?
        xembed = self.embedding(x_)

	# todo: normalize according to reimannian metric
        xembed = F.normalize(xembed, p=2, dim=1)

        yembed = self.embedding(y_)
        yembed = F.normalize(yembed, p=2, dim=1)

        assert (len(xembed) == len(yembed))

        #TODO: insert reimannian metric here as Ay         
	elemprod = xembed * yembed
	dot = torch.sum(elemprod, dim=1)                                                                       

        return dot

# Corpus contains a list of sentences. Each sentence is a list of words
# Data pulled from:
# https://github.com/RaRe-Technologies/gensim-data
CORPUS_NAME="text8"
# We first try to load locally. If this fails, we ask gensim
# to kindly download the data for us.
print ("loading corpus: %s" % (CORPUS_NAME, ))
try:
    print("loading gensim locally...")
    sys.path.insert(0, api.base_dir)
    module = __import__(CORPUS_NAME)
    corpus = module.load_data()
    print("Done.")
except Exception as e:
    print("unable to find text8 locally.\nERROR: %s" % (e, ))
    print("Downloading using gensim-data...")
    corpus = api.load(CORPUS_NAME)
    print("Done.")


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
    # TODO: figure out how to ask for the least loaded cuda device...
    # for now, use the *last* device, since most people will pick up
    # the first devices. So, this should give us a free CUDA card :]
    device = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')
    print("device: %s" % device)

    # TODO: also save optimizer data so we can restart
    if loadpath is not None and os.path.isfile(loadpath):
        print("loading model from %s" % loadpath)
        model = torch.load(loadpath)
        model.eval()
    else:
        print("creating new model...")
        model = Word2Vec(sampler, nhidden=NHIDDEN)

    assert (model)
    print("network: ")
    print(model)
    print("transferring model to device...")
    model.to(device)
    print("done.")

    print("setting up signal handler...")
    def save_model():
        print ("saving model to %s" % savepath)
        torch.save(model, savepath)

    def save_model_handler(signal, frame):
	save_model()
	sys.exit(0)
    signal.signal(signal.SIGTERM, save_model_handler)
    signal.signal(signal.SIGINT, save_model_handler)
    print ("setup signal handlers.")

    # optimise
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    print ("constructed optimizer and criterion.")


    dataset = ConcatDataset([SentenceSkipgramDataset(s, sampler) for s in corpus])
    print("full concat dataset: %s" % dataset)
    print("len of dataset: %s" % len(dataset))

    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    print("constructed dataloader.")

    last_print_time = datetime.datetime.now()
    running_loss = 0
    for epoch in range(EPOCHS):
	if savepath is not None: save_model()
        for i, batch in enumerate(dataloader):
            batch.to(device)
            # TODO: understand why I need to perform this column indexing.
            x_ = batch[:, 0].to(device)
            y_ = batch[:, 1].to(device)
            is_positive = batch[:, 2].to(device)

            # Loss calculation
            optimizer.zero_grad()   # zero the gradient buffers
            # why does this not auto batch?
            y = model(x_, y_)
            # print("is_positive: %s | y: %s " % (is_positive, y))
            loss = criterion(y, is_positive.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            del loss

            now = datetime.datetime.now()
            if i % LOSS_PRINT_STEP == LOSS_PRINT_STEP - 1:    # print every 2000 mini-batches
                print('[epoch:%s, batch:%5d, elems: %5d, loss: %.3f, dt: %s, time:%s]' %
                          (epoch + 1,
                           i + 1,
                           (i + 1) * BATCH_SIZE,
                           running_loss / LOSS_PRINT_STEP, 
                           (now - last_print_time),
                           now.strftime("%X-%a-%b") + ".model"
                           ))
                last_print_time = now
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

