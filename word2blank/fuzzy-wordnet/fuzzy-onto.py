import os
import gensim
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from gensim.models.keyedvectors import KeyedVectors
from collections import OrderedDict

def load_embedding(fpath):
    """
        Using Gensim to load FastText embeddings
    """
    emb = dict()
    wv_from_bin = KeyedVectors.load_word2vec_format(fpath, limit=500000)
    for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
        coefs = np.asarray(vector, dtype='float32')
        if word.lower() not in emb:
            emb[word.lower()] = coefs
    return emb

def normalize(word_vecs, l_dim):
    """
        l_dim(0): sum up the columns
        l_dim(1): sum up the rows
    """
    vec_mat = np.array(list(word_vecs.values()))
    vec_mat = np.exp(vec_mat)                       # e^x for x in all vectors
    vec_mat = preprocessing.normalize(vec_mat, norm='l1', axis=l_dim)
    vecs = np.vsplit(vec_mat, len(word_vecs.keys()))
    word_vecs = zip(list(word_vecs.keys()), vecs)
    return dict(word_vecs)
        

def discretize(word_vecs, l_dim):
    """
        forall x in word_vecs[word]::
            x = 1 if x > threshold
            x = 0 otherwise
        threshold = average
    """
    vec_mat = np.array(list(word_vecs.values())).reshape(len(word_vecs), 300)
    threshold = np.mean(vec_mat, axis=l_dim)
    vec_mat = (vec_mat >= threshold) * 1
    vecs = np.vsplit(vec_mat, len(word_vecs.keys()))
    word_vecs = dict(zip(list(word_vecs.keys()), vecs))
    word_vecs['<TOP>'] = np.ones(300, dtype=int)
    word_vecs['<BOT>'] = np.zeros(300, dtype=int)
    return word_vecs

def construct_ontology(word_vecs):
    pass

def similarity(word_vecs, word):
    vec = word_vecs[word]
    sim = dict()
    for w in word_vecs:
        sim[(word, w)] = int(np.reshape(np.dot(vec, np.transpose(word_vecs[w])), 1)[0])
    return sim


if __name__ == '__main__':
    dirname = '/scratch/alokdebnath/MODELS/'
    fname = 'wiki-news-300d-1M.bin'
    NDIMS = 300
    word_vecs = load_embedding(os.path.join(dirname, fname))
    word_vecs = normalize(word_vecs, 0)
    word_vecs = discretize(word_vecs, 0)
    sims = similarity(word_vecs, 'nepotism')
    dd = OrderedDict(sorted(sims.items(), key=lambda x: x[1], reverse=True))
    print(list(dd.items())[:10])
