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
from tabulate import tabulate
import numpy as np
import signal
import sys
import click
import datetime 
import os
import math
 
# https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.ipynb

torch.manual_seed(1)

LEARNING_RATE=1000
LOSS_PRINT_NBATCHES = 5
MODEL_SAVE_NBATCHES = 200
BATCH_SIZE = 1024
EPOCHS = 1000
NHIDDEN = 100
WINDOW_SIZE=2

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

def DEFAULT_MODELPATH():
    now = datetime.datetime.now()
    return now.strftime("%X-%a-%b") + ".model"

# Compute frequencies of words in the given sentence ws, and the current
# word frequences wfs
def update_wfs(ws, wfs):
    """ws: iterable, wfs: Counter"""
    for w in ws:
        if w in STOPWORDS: continue
        wfs[w] += 1

def plot_wfs(wfs):
    freqs = [math.log(val / ftot) for val in wfs.values()]
    freqs.sort(reverse=True)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(range(len(freqs)), freqs)
    plt.savefig("wordfrequency.png")

# http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
# Sampler to sample from given frequency distribution
class Sampler:
    def __init__(self, wfs):
        POW = 0.75 # recommended power to raise to 

        # TODO: refactor code
        ftot = float(sum(wfs.values()))
        MAX_FREQ = max(wfs.values()) / ftot
        MIN_FREQ = min(wfs.values()) / ftot
        # 80/20: Throw stuff that less that 20% from the bottom.
        THROW_BELOW_FREQ = MIN_FREQ * 1.2
        print("max freq: %s | min freq %s | freq below thrown: %s" % (MAX_FREQ, MIN_FREQ, THROW_BELOW_FREQ))

        # plot the wfs
        plot_wfs(wfs)
        
        # TODO: add more stopwords
        todelete = set()
        for w in wfs: 
            if wfs[w] / ftot <= THROW_BELOW_FREQ: todelete.add(w)

        print("#words thrown away: %s | thrown/total ratio: %s" % 
              (len(todelete), float(len(todelete)) / len(wfs)))
        # delete words that occur with way less frequency
        for w in todelete: del wfs[w]
        print("#words left over: %s" % (len(wfs)))

        # now that we have thrown stuff away, revise total frequency
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

    def wordprob(self, w):
        if w in self.ws2ix:
            return self.probs[self.ws2ix[w]]
        return 0

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



def mk_onehot(sampler, w):
    # v = torch.zeros(len(sampler)).float()
    #v[sampler.wordix(w)] = 1.0
    # return v

    # TODO: understand why this does not work
    # return Variable(torch.LongTensor([sampler.wordix(w)], device=device))
    return Variable(torch.LongTensor([sampler.wordix(w)]))

# probability of keeping a word
def calc_prob_keep_word(w, sampler):
    # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    T = 0.001
    z = sampler.wordprob(w)
    if z == 0: return 0

    prob =  ((z / T) ** 0.5 + 1) * (T / (z + T))
    return prob

class SentenceSkipgramDataset(Dataset):
    def __init__(self, s, sampler, window_size):
        """
        sampler: sampler for negative sampling
        wfs: word frequencies
        window_size: window size
        """
        # SUBSAMPLING
        self.s =[]
        rand = torch.rand(len(s))
        for i, w in enumerate(s):
            if rand[i] < calc_prob_keep_word(w, sampler):
                self.s.append(w)

        self.sampler = sampler
        self.window_size = window_size
        self.NNEGATIVES_PER_WORD = 4
        self.NNEGATIVES = len(self.s) * self.NNEGATIVES_PER_WORD
        # for every word that is in the context [window_size, len(self.s) - window_size],
        # we have (window_size * 2) elements
        # idx / (WINDOW_SIZE * 2) -> word to pick
        # (idx % (WINDOW_SIZE * 2) - WINDOW_SIZE) -> position wrt context
        self.NPOSITIVES = (len(self.s) - 2 * window_size) * (window_size * 2)

    def __getitem__(self, idx):
        # move index by 1 so we are in range [1..len - 2]
        # TODO: generalize this to a window.
        if (idx >= self.NPOSITIVES):
             idx -= self.NPOSITIVES
             idx = idx / self.NNEGATIVES_PER_WORD
             x_ = self.s[idx]
             y_ = sampler.sample()
             is_positive_ = 0
        else:
            base_idx = idx // (self.window_size * 2)
            base_idx += self.window_size

            sign =  1 if (idx % 2) == 0 else (-1)
            delta = (idx % (self.window_size)) * sign
            assert(0 <= base_idx < len(self.s))
            assert(0 <= base_idx + delta < len(self.s))

            (x_, y_, is_positive_) = (self.s[base_idx], self.s[base_idx + delta], 1)

        x_ = mk_onehot(sampler, x_)
        y_ = mk_onehot(sampler, y_)
        is_positive_ = Variable(torch.LongTensor([is_positive_]))
        out =  torch.cat([x_, y_, is_positive_])
        return out

    def __len__(self):
        # we can't query the first or last value.
        # first because it has no left, last because it has no right
        return self.NPOSITIVES + self.NNEGATIVES



def cosine_similarity_batched_vec(xs, ys):
    """Return the cosine similarity of each indivisual xs[i] with ys[i]"""
    xs = F.normalize(xs, p=2, dim=1)
    ys = F.normalize(ys, p=2, dim=1)
    assert (len(xs) == len(ys))
    #TODO: insert reimannian metric here as Ay         
    elemprod = xs * ys
    dot = torch.sum(elemprod, dim=1)

    # as done in:
    # https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.ipynb
    #dot = F.logsigmoid(dot)
    # dot = F.sigmoid(dot)

    return dot

# Word2Vec word2vec
# https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.ipynb
class Word2Vec(nn.Module):
    def __init__(self, sampler, nhidden):
        super(Word2Vec, self).__init__()
        self.nhidden = nhidden
        nwords = len(sampler)
        """nwords: number of words"""
        self.embedding = nn.Embedding(len(sampler), nhidden)

    # run the forward pass which matches word y in the context of x
    def forward(self, x_, y_):
        # 2d vectors of data x batch_size
        # TODO: is batch_size outer or inner dim?
        xembed = self.embedding(x_)
        yembed = self.embedding(y_)

        return cosine_similarity_batched_vec(xembed, yembed)

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
for ws in corpus:
    update_wfs(ws, wfs)

sampler = Sampler(wfs)
print ("Done.")


# setup device
device = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')
print("device: %s" % device)




@click.group()
def cli():
    pass

def torch_status_dump():
    print("---")
    print("CUDA available?: %s" % torch.cuda.is_available())
    print("GPU device: |%s|" % torch.cuda.get_device_name(0))
    print("---")

def eval_model_on_common_words(sampler, model, device):
    word_pairs = [("sensory", "system"), ("situations", "development"), ("workers", "system")]
    headers = ["word1", "word2", "similarity"]
    table = []
    for (x, y) in word_pairs:
        x_ = mk_onehot(sampler, x).to(device)
        y_ = mk_onehot(sampler, y).to(device)
        with torch.no_grad():
            sim = model(x_, y_)
            table.append((x, y, sim))
    print(tabulate(table, headers))

    



@click.command()
@click.option('--savepath', default=DEFAULT_MODELPATH(), help='Path to save model')
@click.option('--loadpath', default=None, help='Path to load model from')
def train(savepath, loadpath):
    # TODO: figure out how to ask for the least loaded cuda device...
    # for now, use the *last* device, since most people will pick up
    # the first devices. So, this should give us a free CUDA card :]
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

    def save_model():
        print ("saving model to %s" % savepath)
        torch.save(model, savepath)

    def save_model_handler(signal, frame):
	save_model()
	sys.exit(0)

    # disable saving on C-c, it seems unstable and prone to destroying the
    # model...
    #signal.signal(signal.SIGTERM, save_model_handler)
    #signal.signal(signal.SIGINT, save_model_handler)
    #print ("setup signal handlers.")

    # optimise
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    print ("constructed optimizer and criterion.")


    print("constructing dataset...")
    dataset = ConcatDataset([SentenceSkipgramDataset(s, sampler, WINDOW_SIZE) for s in corpus])
    print("Done. number of samples in dataset: %s" % len(dataset))

    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    print("constructed dataloader.")

    last_print_time = datetime.datetime.now()
    running_loss = 0
    print("starting training. time: %s" % (last_print_time))
    for epoch in range(EPOCHS):
	if savepath is not None: save_model()
        for i, batch in enumerate(dataloader):
            # batch.to(device)
            # TODO: understand why I need to perform this column indexing.
            x_ = batch[:, 0].to(device)
            y_ = batch[:, 1].to(device)
            is_positive = batch[:, 2].to(device)

            # why does this not auto batch?
            y = model(x_, y_)
            # Loss calculation
            loss = criterion(y, is_positive.float())
            # Step
            optimizer.zero_grad()   # zero the gradient buffers
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            del loss

            now = datetime.datetime.now()
            if i % LOSS_PRINT_NBATCHES == LOSS_PRINT_NBATCHES - 1:    # print every 2000 mini-batches
                print('[epoch:%s, batch:%5d, elems: %5d, loss(per item): %.3f, dt: %s, time:%s]' %
                          (epoch + 1,
                           i + 1,
                           (i + 1) * BATCH_SIZE,
                           running_loss / (LOSS_PRINT_NBATCHES * BATCH_SIZE), 
                           (now - last_print_time),
                           now.strftime("%X-%a-%b") + ".model"
                           ))
                last_print_time = now
                running_loss = 0.0
                eval_model_on_common_words(sampler, model, device)

            if i % MODEL_SAVE_NBATCHES == MODEL_SAVE_NBATCHES - 1:
                save_model()
    # save the model at the end of the training run
    save_model()


@click.command()
@click.argument('modelpath')
def test(modelpath):
    model = torch.load(modelpath)
    model.eval()
    criterion = nn.MSELoss()
 
    # list of all words
    WORDS = sampler.words()

    def find_sim(w1, w2):
        with torch.no_grad():
            vcur = mk_onehot(sampler, w1).to(device)
            vother = mk_onehot(sampler, w2).to(device)
            return model(vcur, vother)[0]
 
    # find closest vectors to each word in the model
    with torch.no_grad():
        for wcur in WORDS[:50]:
            ws = []
            print ("* %s " % (wcur, ))
            for wother in WORDS:
                ws.append((wother, find_sim(wcur, wother)))
            # pick ascending and descding words, so top 10 and bottom 5
            wsout = []
            ws.sort(key=lambda wd: wd[1], reverse=True)
            wsout = ws [:10]

            ws.sort(key=lambda wd: wd[1])
            wsout += ws [:5]
            print ("\n".join(["\t%s -> %s" % (w, d) for (w, d) in wsout]))
    # drop people in a terminal at this point.
    import pudb; pudb.set_trace()
 
cli.add_command(train)
cli.add_command(test)

if __name__ == "__main__":
    torch_status_dump()
    cli()

