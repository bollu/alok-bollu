import os
import gensim
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from gensim.models.keyedvectors import KeyedVectors
from collections import OrderedDict

def load_embedding(fpath, VOCAB):
    """
        Using Gensim to load FastText embeddings
    """
    emb = dict()
    wv_from_bin = KeyedVectors.load_word2vec_format(fpath, limit=VOCAB)
    for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
        coefs = np.asarray(vector, dtype='float32')
        if word not in emb:
            emb[word] = coefs
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
        

def discretize(word_vecs, l_dim, NDIMS):
    """
        forall x in word_vecs[word]::
            x = 1 if x > threshold
            x = 0 otherwise
        threshold = average
    """
    vec_mat = np.array(list(word_vecs.values())).reshape(len(word_vecs), NDIMS)
    threshold = np.mean(vec_mat, axis=l_dim)
    vec_mat = (vec_mat >= threshold) * 1
    vecs = np.vsplit(vec_mat, len(word_vecs.keys()))
    word_vecs = dict(zip(list(word_vecs.keys()), vecs))
    word_vecs['<TOP>'] = np.ones(NDIMS, dtype=int)
    word_vecs['<BOT>'] = np.zeros(NDIMS, dtype=int)
    return word_vecs

def decode(word_vecs, vec):
    """
        Given a vector which is not in the dictionary, look-up those words which are closest to it, excluding <TOP>
    """
    sim = 0
    word = str()
    for w in word_vecs:
        if int(np.reshape(np.dot(vec, np.transpose(word_vecs[w])), 1)[0]) > sim and w != '<TOP>' and w != '<BOT>':
            word = w
            sim = int(np.reshape(np.dot(vec, np.transpose(word_vecs[w])), 1)[0])
    return word


def topn_similarity(word_vecs, word, n):
    """
        Given embedding dictionary and word
        find the top n words most similar to this word
        weight of the similarity determined by dot product
    """
    vec = word_vecs[word]
    sim = dict()
    for w in word_vecs:
        if w != '<TOP>' and w != '<BOT>':
            sim[w] = int(np.reshape(np.dot(vec, np.transpose(word_vecs[w])), 1)[0])
    dd = OrderedDict(sorted(sim.items(), key=lambda x: x[1], reverse=True))
    return list(dd.keys())[:n]

def union(emb, w1, w2):
    return decode(emb, np.absolute(emb[w1] + emb[w2]  - emb[w1] * emb[w2]))

def intersection(emb, w1, w2):
    return decode(emb, (emb[w1] * emb[w2]))

def difference(emb, w1, w2):
    return decode(emb, np.absolute(emb[w1] - emb[w2]))

def logicaland(emb, w1, w2):
    return decode(emb, np.logical_and(emb[w1], emb[w2]))

def logicalor(emb, w1, w2):
    return decode(emb, np.logical_or(emb[w1], emb[w2]))

def logicalxor(emb, w1, w2):
    return decode(emb, np.logical_xor(emb[w1], emb[w2]))

def constructgraph(emb):
    pass

if __name__ == '__main__':
    dirname = '../MODELS/'
    fname = 'gensim_glove_vectors.txt'
    NDIMS = 300
    VOCAB = 50000
    nsim = 10
    word_vecs = load_embedding(os.path.join(dirname, fname), VOCAB)
    word_vecs = normalize(word_vecs, 0)
    word_vecs = discretize(word_vecs, 0, NDIMS)
    for w in word_vecs:
        word_vecs[w] = np.reshape(word_vecs[w], NDIMS)

    w1 = 'view'
    w2 = 'idea'

    print('sim({})'.format(w1), end='\t')
    print(topn_similarity(word_vecs, w1, nsim))

    print('sim({})'.format(w2), end='\t')
    print(topn_similarity(word_vecs, w2, nsim))
    
    print('{} u {}'.format(w1, w2), end='\t')
    print(union(word_vecs, w1, w2))
    
    print('{} n {}'.format(w1, w2), end='\t')
    print(intersection(word_vecs, w1, w2))

    print('{} - {}'.format(w1, w2), end='\t')
    print(difference(word_vecs, w1, w2))

    print('{} AND {}'.format(w1, w2), end='\t')
    print(logicaland(word_vecs, w1, w2))

    print('{} OR {}'.format(w1, w2), end='\t')
    print(logicalor(word_vecs, w1, w2))

    print('{} XOR {}'.format(w1, w2), end='\t')
    print(logicalxor(word_vecs, w1, w2))
