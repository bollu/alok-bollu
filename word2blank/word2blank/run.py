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
from prompt_toolkit.completion import WordCompleter, ThreadedCompleter
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit import print_formatted_text
import prompt_toolkit.shortcuts
import sacred
import sacred.observers
import progressbar
import pudb


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
        print("unable to find text8 locally.\nERROR: %s" % (e, ))
        print("Downloading using gensim-data...")
        corpus = api.load(CORPUS_NAME)
        print("Done.")

    corpus = list(corpus)
    corpus = flatten(corpus)
    print("number of words in corpus (original): %s" % (len(corpus), ))

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


def dot(v, w, metric):
    return dots(v.view(1, -1), w.view(1, -1), metric)

    
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

def normalize(vs, metric):
    # vs = [S1 x EMBEDSIZE]
    # metric = [EMBEDSIZE x EMBEDSIZE]
    # normvs = [S1 x EMBEDSIZE]
    normvs = torch.zeros(vs.size()).to(DEVICE)
    BATCHSIZE = 512
    with prompt_toolkit.shortcuts.ProgressBar() as pb:
        for i in pb(range(math.ceil(vs.size()[0] / BATCHSIZE))):
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
    ERROR_THRESHOLD = 0.1
    return normvs


class Word2Man(nn.Module):
    def __init__(self, VOCABSIZE, EMBEDSIZE, DEVICE):
        super(Word2Man, self).__init__()
        LOGGER.start("creating EMBEDM")
        self.EMBEDM = nn.Parameter(Variable(torch.randn(VOCABSIZE, EMBEDSIZE).to(DEVICE), requires_grad=True))
        LOGGER.end()

        LOGGER.start("creating METRIC")
        # TODO: change this so we can have any nonzero symmetric matrix.
        # add loss so that we don't allow zero matrix. That should be
        # nondegenrate quadratic form.
        self.METRIC_SQRT = nn.Parameter(torch.randn([EMBEDSIZE, EMBEDSIZE]).to(DEVICE), requires_grad=True)
        self.METRIC = torch.mm(self.METRIC_SQRT, self.METRIC_SQRT.t())
        LOGGER.end()

    def forward(self, xs):
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
        # recompute metric again
        self.METRIC = torch.mm(self.METRIC_SQRT, self.METRIC_SQRT.t())

        xsembeds_dots_embeds = dots(xsembeds, self.EMBEDM, self.METRIC)
        # TODO: why is this correct? I don't geddit.
        # what in the fuck does it mean to log softmax cosine?
        # [BATCHSIZE x VOCABSIZE]
        xsembeds_dots_embeds = F.log_softmax(xsembeds_dots_embeds, dim=1)

        return xsembeds_dots_embeds

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

        self._init_from_params()


    def _init_from_params(self):
        TEXT = load_corpus(LOGGER, self.NWORDS)

        LOGGER.start("building vocabulary")
        VOCAB = set(TEXT)
        VOCABSIZE = len(VOCAB)
        LOGGER.end()

        LOGGER.start("creating word2man")
        self.WORD2MAN = Word2Man(VOCABSIZE, self.EMBEDSIZE, DEVICE)
        LOGGER.end()

        LOGGER.start("creating OPTIMISER")
        self.optimizer = optim.Adam(self.WORD2MAN.parameters(), lr=self.LEARNING_RATE)
        LOGGER.end()


        LOGGER.start("creating dataset...")
        self.DATASET = SkipGramOneHotDataset(LOGGER, TEXT, VOCAB, VOCABSIZE, self.WINDOWSIZE)
        LOGGER.end()

        # TODO: pytorch dataloader is sad since it doesn't save state.
        # make a version that does save state.
        LOGGER.start("creating DATA\n")
        self.DATA = DataLoader(self.DATASET, batch_size=self.BATCHSIZE, shuffle=True)
        LOGGER.end()

    # does not work :(
    # def __getstate__(self):
    #     return {
    #         "EPOCHS": self.EPOCHS,
    #         "BATCHSIZE": self.BATCHSIZE,
    #         "EMBEDSIZE": self.EMBEDSIZE,
    #         "LEARNINGRATE": self.LEARNING_RATE,
    #         "WINDOWSIZE": self.WINDOWSIZE,
    #         "NWORDS": self.NWORDS,
    #         "CREATE_TIME": self.create_time,
    #         "WORD2MAN": self.WORD2MAN.state_dict(),
    #         "OPTIMIZER": self.optimizer.state_dict()
    #     }
    # 
    # def __setstate__(self, state):
    #     self.EPOCHS = state["EPOCHS"]
    #     self.BATCHSIZE = state["BATCHSIZE"]
    #     self.EMBEDSIZE = state["EMBEDSIZE"]
    #     self.LEARNING_RATE = state["LEARNINGRATE"]
    #     self.WINDOWSIZE = state["WINDOWSIZE"]
    #     self.NWORDS = state["NWORDS"]
    #     self.create_time = state["CREATE_TIME"]

    #     self._init_from_params()
    #     self.WORD2MAN.load_state_dict(state["WORD2MAN"])
    #     self.WORD2MAN.eval()
    #     self.optimizer.load_state_dict(state["OPTIMIZER"])

    #     print("on loading:")
    #     print("METRIC:\n%s" % (self.WORD2MAN.METRIC, ))
    #     print("EMBEDM:\n%s" % (self.WORD2MAN.EMBEDM, ))



def mk_word_histogram(ws, vocab):
    """count frequency of words in words, given vocabulary size."""
    w2f = { w : 0 for w in vocab }
    for w in ws:
        w2f[w] += 1
    return w2f

class SkipGramNHotDataset(Dataset):
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
        ix += self.WINDOWSIZE
        focusix = [self.TEXT[ix]]
        ctxixs = [self.TEXT[ix + deltaix] for deltaix in range(-self.WINDOWSIZE, self.WINDOWSIZE + 1)]

        return {'focusix': hot(focusix, self.W2I, self.VOCABSIZE),
                'ctxixs': hot(ctxixs, self.W2I, self.VOCABSIZE)}

    def __len__(self):
        # we can't query the first or last value.
        # first because it has no left, last because it has no right
        return (len(self.TEXT) - 2 * self.WINDOWSIZE) 

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

        return {'focusonehot': hot([self.TEXT[focusix]], self.W2I, self.VOCABSIZE),
                'ctxtruelabel': self.W2I[self.TEXT[focusix + deltaix]] 
                }

    def __len__(self):
        # we can't query the first or last value.
        # first because it has no left, last because it has no right
        return (len(self.TEXT) - 2 * self.WINDOWSIZE) * (2 * self.WINDOWSIZE)

def hot(ws, W2I, VOCABSIZE):
    """
    hot *normalized* vector for each word in ws
    """
    v = Variable(torch.zeros(VOCABSIZE).float())
    for w in ws: v[W2I[w]] = 1.0
    return v / float(len(ws))


def cli_prompt():
    """Call to launch prompt interface."""

    LOGGER.start("normalizing EMBEDM...")
    PARAMS.WORD2MAN.EMBEDM = PARAMS.WORD2MAN.EMBEDM.to(DEVICE)
    PARAMS.WORD2MAN.METRIC = PARAMS.WORD2MAN.METRIC.to(DEVICE)
    EMBEDNORM = normalize(PARAMS.WORD2MAN.EMBEDM, PARAMS.WORD2MAN.METRIC).to(DEVICE)
    LOGGER.end("done.")


    def word_to_embed_vector(w):
        """
        find the embedded vector of word w
        returns: [1 x EMBEDSIZE]
        """
        v = PARAMS.WORD2MAN.EMBEDM[PARAMS.DATASET.W2I[w], :]
        return normalize(v.view(1, -1), PARAMS.WORD2MAN.METRIC)

    def test_find_close_vectors(v):
        """ Find vectors close to w in the normalized embedding"""
        # dot [1 x EMBEDSIZE] [VOCABSIZE x EMBEDSIZE] = [1 x VOCABSIZE]
        wix2sim = dots(v, EMBEDNORM, PARAMS.WORD2MAN.METRIC)

        wordweights = [(PARAMS.DATASET.I2W[i], wix2sim[0][i].item()) for i in range(PARAMS.DATASET.VOCABSIZE)]
        # The story of floats which cannot be ordered. I found
        # out I need this filter once I found this entry in the
        # table: ('classified', nan)
        wordweights = list(filter(lambda ww: not math.isnan(ww[1]), wordweights))
        # sort AFTER removing NaNs
        wordweights.sort(key=lambda wdot: wdot[1], reverse=True)

        return wordweights


    def prompt_word(session):
        """Prompt for a word and print the closest vectors to the word"""
        # [VOCABSIZE x EMBEDSIZE]
        COMPLETER = ThreadedCompleter(WordCompleter(PARAMS.DATASET.VOCAB))

        raw = session.prompt("type in command>", completer=COMPLETER).split() 
        # raw = session.prompt("type in command>", completer=COMPLETER).split()
        if len(raw) == 0:
            return
        if raw[0] == "help" or raw[0] == "?":
            print_formatted_text("near <word> | sim <w1> <w2> <w3> | dot <w1> <w2> | metric | debug")
            return
        elif raw[0] == "debug":
            pudb.set_trace()
        elif raw[0] == "near" or raw[0] == "neighbour":
            if len(raw) != 2:
                print("error: expected near <w>")
                return
            wordweights = test_find_close_vectors(word_to_embed_vector(raw[1]))
            for (word, weight) in wordweights[:15]:
                print_formatted_text("\t%s: %s" % (word, weight))
        elif raw[0] == "sim":
            if len(raw) != 4:
                print("error: expected sim <w1> <w2> <w3>")
                return
            v1 = word_to_embed_vector(raw[1])
            v2 = word_to_embed_vector(raw[2])
            v3 = word_to_embed_vector(raw[3])
            vsim = normalize(v2 - v1 + v3, PARAMS.WORD2MAN.METRIC) 
            wordweights = test_find_close_vectors(vsim)
            for (word, weight) in wordweights[:15]:
                print_formatted_text("\t%s: %s" % (word, weight))
        elif raw[0] == "dot":
            if len(raw) != 3:
                print("error: expected dot <w1> <w2>")
                return

            v1 = normalize(word_to_embed_vector(raw[1]), PARAMS.WORD2MAN.METRIC)
            v2 = normalize(word_to_embed_vector(raw[2]), PARAMS.WORD2MAN.METRIC)
            print_formatted_text("\t%s" % (dots(v1, v2, PARAMS.WORD2MAN.METRIC), ))
        elif raw[0] == "metric":
            print(PARAMS.WORD2MAN.METRIC)
            w,v = torch.eig(PARAMS.WORD2MAN.METRIC,eigenvectors=True)
            print_formatted_text("eigenvalues:\n%s" % (w, ))
            print_formatted_text("eigenvectors:\n%s" % (v, ))
            (s, v, d) = torch.svd(PARAMS.WORD2MAN.METRIC)
            print("SVD :=\n%s\n%s\n%s" % (s, v, d))
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
DEVICE = torch.device(torch.cuda.device_count() - 1) if torch.cuda.is_available() else torch.device('cpu')
LOGGER.end("device: %s" % DEVICE)


PARAMS = None
if PARSED.loadpath is not None:
    LOGGER.start("loading model from: %s" % (PARSED.loadpath))
    # pass the device so that the tensors live on the correct device.
    # this might be stale since we were on a different device before.
    try:
        PARAMS = torch.load(PARSED.loadpath, map_location=DEVICE)
        LOGGER.end("loaded params from: %s" % PARAMS.create_time)
    except FileNotFoundError as e:
        LOGGER.end("file (%s) not found. Creating new model" % (PARSED.loadpath, ))
        
if PARAMS is None:
    LOGGER.start("Creating new parameters")
    PARAMS = Parameters(LOGGER, DEVICE)
    LOGGER.end()

EXPERIMENT = sacred.Experiment()
EXPERIMENT.add_config(EPOCHS = PARAMS.EPOCHS,
                      BATCHSIZE = PARAMS.BATCHSIZE,
                      EMBEDSIZE = PARAMS.EMBEDSIZE,
                      LEARNING_RATE = PARAMS.LEARNING_RATE,
                      WINDOWSIZE = PARAMS.WINDOWSIZE,
                      NWORDS = PARAMS.NWORDS)
EXPERIMENT.observers.append(sacred.observers.FileStorageObserver.create('runs'))


# @EXPERIMENT.capture
def traincli(savepath):
    global PARAMS
    def save():
        LOGGER.start("\nsaving model to: %s" % (savepath))
        with open(savepath, "wb") as sf:
            torch.save(PARAMS, sf)
        # EXPERIMENT.add_artifact(savepath)
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
    time_last_save = datetime.datetime.now()
    time_last_print = datetime.datetime.now()
    last_print_ix = 0
    for epoch in range(PARAMS.EPOCHS):
        for train in PARAMS.DATA:
            ix += 1
            # [BATCHSIZE x VOCABSIZE]
            xs = train['focusonehot'].to(DEVICE)
            # [BATCHSIZE], contains target label per batch
            target_labels = train['ctxtruelabel'].to(DEVICE)

            PARAMS.optimizer.zero_grad()   # zero the gradient buffers

            # dot product of the embedding of the hidden xs vector with 
            # every other hidden vector
            # xs_dots_embeds: [BATCSIZE x VOCABSIZE]
            xs_dots_embeds = PARAMS.WORD2MAN(xs)

            l = F.nll_loss(xs_dots_embeds, target_labels)
            loss_sum += l.item()
            # l.backward(retain_graph=True)
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
            TARGET_SAVE_TIME_IN_S = 60 * 15 # save every 15 minutes
            if (now - time_last_save).seconds > TARGET_SAVE_TIME_IN_S:
                save()
                time_last_save = now
    save()





# @EXPERIMENT.main
def main():
    if PARSED.command == "train":
        traincli(PARSED.savepath)
    elif PARSED.command == "test":
        cli_prompt()
    else:
        raise RuntimeError("unknown command: %s" % PARSED.command)

if __name__ == "__main__":
    main()
    # EXPERIMENT.run()

