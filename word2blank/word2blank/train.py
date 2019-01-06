#!/usr/bin/env python
import gensim.downloader as api

corpus = api.load('text8')  # download the corpus and return it opened as an iterable
print(len(list(corpus)))
