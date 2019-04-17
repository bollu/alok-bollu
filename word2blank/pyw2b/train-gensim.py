#!/usr/bin/env python3
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import sys
import argparse
import datetime
import gensim.downloader as api

def current_time_str():
    """Return the current time as a string"""
    return datetime.datetime.now().strftime("%X-%a-%b")

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
LOGGER = TimeLogger()

def parse(s):
    def DEFAULT_MODELPATH():
        now = datetime.datetime.now()
        return "word2vec-gensim.model"
        # return "save-auto-%s.model" % (current_time_str(), )
    p = argparse.ArgumentParser()

    p.add_argument("--loadpath", help="path to model file to load from", default=None)
    p.add_argument("--savepath", help="path to save model to", default=DEFAULT_MODELPATH())
    
    return p.parse_args(s)


def load_corpus():
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
    LOGGER.log("number of words in corpus (original): %s" % (len(corpus), ))
    LOGGER.end()
    return corpus


def train(savepath, loadpath):
    corpus = load_corpus()
    # LOGGER.start("loading text8")
    # corpus = Text8Corpus("text8")
    # LOGGER.end()

    LOGGER.start("creating model")
    if loadpath is not None:
        model = Word2Vec.load("loadpath")
    else:
        model = Word2Vec(size=300, window=5, workers=16, min_count=2, sg=1)
    LOGGER.end()
    LOGGER.start("building vocab")
    model.build_vocab(corpus)
    LOGGER.end()

    LOGGER.start("starting training")
    model.train(corpus, total_examples=len(corpus), epochs=15, report_delay=1)
    LOGGER.end()
    model.save(savepath)

if __name__ == "__main__":
    PARSED = parse(sys.argv[1:])
    train(PARSED.savepath, PARSED.loadpath)
