import os
import gensim
import re
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from scipy import spatial
from collections import OrderedDict
#from numba import jit, cuda

def load_embedding(fpath, VOCAB):
    """
        Using Gensim to load FastText embeddings
    """
    print("Loading embeddings...")
    emb = dict()
    wv_from_bin = KeyedVectors.load_word2vec_format(fpath, limit=VOCAB)
    for word, vector in tqdm(zip(wv_from_bin.vocab, wv_from_bin.vectors)):
        coefs = np.asarray(vector, dtype='float32')
        if word not in emb:
            emb[word] = coefs
    return emb

def normalize(word_vecs, axis, NDIMS):
    """
        axis(0): sum up the columns
        axis(1): sum up the rows
    """
    print("Normalizing across axis = %d" % axis)
    vec_mat = np.array(list(word_vecs.values()))
    vec_mat = np.exp(vec_mat)                       # e^x for x in all vectors
    vec_mat = preprocessing.normalize(vec_mat, norm='l1', axis=axis)
    vecs = np.vsplit(vec_mat, len(word_vecs.keys()))
    word_vecs = dict(zip(list(word_vecs.keys()), [v[0] for v in vecs]))
    return word_vecs
        

def discretize(word_vecs, axis, NDIMS):
    """
        forall x in word_vecs[word]::
            x = 1 if x > threshold
            x = 0 otherwise
        threshold = average
    """
    print("Discretizing across = %d" % axis)
    vec_mat = np.array(list(word_vecs.values()))
    threshold = np.mean(vec_mat, axis=axis)
    vec_mat = (vec_mat >= threshold) * 1
    vecs = np.vsplit(vec_mat, len(word_vecs.keys()))
    word_vecs = dict(zip(list(word_vecs.keys()), [v[0] for v in vecs]))
    # word_vecs['<TOP>'] = np.ones(NDIMS, dtype=int)
    # word_vecs['<BOT>'] = np.zeros(NDIMS, dtype=int)
    return word_vecs

def gensim_store(fname, embs, VOCAB, NDIMS):
    """
        Ref :https://github.com/RaRe-Technologies/gensim/blob/3d2227d58b10d0493006a3d7e63b98d64e991e60/gensim/models/keyedvectors.py#L130
        Write out the discretized word representations into a file, so that we can load more of them
        `fname` : the file used to save the vectors in
        `embs`: word embeddings in dictionary format
        `VOCAB` : number of words
        `NDIMS` : number of dimensions
    """
    print("Storing vectors")
    g = open(fname, 'wb+')
    g.write(gensim.utils.to_utf8('%s %s\n' % (VOCAB, NDIMS)))
    for word, vector in tqdm(embs.items()):
        g.write(gensim.utils.to_utf8('%s %s\n' % (word, ' '.join('%f' % val for val in vector))))
    g.close()
    return


def decode(word_vecs, vec):
    """
        Given a vector which is not in the dictionary, look-up those words which are closest to it, excluding <TOP>
    """
    sim = -1000
    word = str()
    for w in word_vecs:
        if 1 - spatial.distance.cosine(vec, word_vecs[w]) > sim and w != '<TOP>' and w != '<BOT>':
            word = w
            sim = 1 - spatial.distance.cosine(vec, word_vecs[w]) 
    return word

def mod(vec):
    return np.sqrt(np.sum(np.square(vec)))


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
            # sim[w] = np.dot(vec, np.transpose(word_vecs[w]))
            sim[w] = 1 - spatial.distance.cosine(vec, word_vecs[w])
            #  sim[w] = np.dot(vec, np.transpose(word_vecs[w]))/(mod(vec)*mod(np.transpose(word_vecs[w])))
    dd = OrderedDict(sorted(sim.items(), key=lambda x: x[1], reverse=True))
    return list(dd.items())[1:n+1]


def thresh_similarity(word_vecs, word, thresh):
    """
        Given embedding dictionary and word
        find words having similarity to this word above thresh level
        weight of the similarity determined by dot product
    """
    vec = word_vecs[word]
    sim = dict()
    for w in word_vecs:
        if w != '<TOP>' and w != '<BOT>':
            sim_val = 1 - spatial.distance.cosine(vec, word_vecs[w])
            if sim_val > thresh:
                sim[w] = sim_val
    return list(sim.keys())

def union(emb, w1, w2):
    return np.absolute(emb[w1] + emb[w2]  - emb[w1] * emb[w2])

def intersection(emb, w1, w2):
    return emb[w1] * emb[w2]

def difference(emb, w1, w2):
    return np.absolute(emb[w1] - emb[w2])

def logicaland(emb, w1, w2):
    return np.logical_and(emb[w1], emb[w2])

def logicalor(emb, w1, w2):
    return np.logical_or(emb[w1], emb[w2])

def logicalxor(emb, w1, w2):
    return np.logical_xor(emb[w1], emb[w2])

if __name__ == '__main__':
    dirname = '../../utilities/MODELS/'
    fname = 'wiki-news-300d-1M.bin'
    # NDIMS = 300
    VOCAB = 999994
    nsim = 10
    # word_vecs = load_embedding(os.path.join(dirname, fname), VOCAB)
    # word_vecs = normalize(word_vecs, 0, NDIMS)
    # word_vecs = discretize(word_vecs, 0, NDIMS)
    # gensim_store('test.bin', word_vecs, VOCAB, NDIMS)
    w1 = 'condition'
    w2 = 'relation'
    disc_vecs = KeyedVectors.load_word2vec_format("test.bin", binary=False) 
    word_vecs = load_embedding("test.bin", VOCAB) 
    
    print('sim({})'.format(w1), end='\t')
    print(disc_vecs.similar_by_word(w1, topn=nsim, restrict_vocab=False))
    
    print('sim({})'.format(w2), end='\t')
    print(disc_vecs.similar_by_word(w2, topn=nsim, restrict_vocab=False))
    
    print('{} u {}'.format(w1, w2), end='\t')
    print(disc_vecs.similar_by_vector(union(word_vecs, w1, w2), topn=1))

    print('{} n {}'.format(w1, w2), end='\t')
    print(disc_vecs.similar_by_vector(intersection(word_vecs, w1, w2), topn=1))

    print('{} - {}'.format(w1, w2), end='\t')
    print(disc_vecs.similar_by_vector(difference(word_vecs, w1, w2), topn=1))

    print('{} OR {}'.format(w1, w2), end='\t')
    print(disc_vecs.similar_by_vector(logicaland(word_vecs, w1, w2), topn=1))
    
    print('{} AND {}'.format(w1, w2), end='\t')
    print(disc_vecs.similar_by_vector(logicalor(word_vecs, w1, w2), topn=1))

    print('{} XOR {}'.format(w1, w2), end='\t')
    print(disc_vecs.similar_by_vector(logicalxor(word_vecs, w1, w2), topn=1))
