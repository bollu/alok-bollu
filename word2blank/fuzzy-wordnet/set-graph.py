import os
from scipy import spatial
import numpy as np
import preprocess
from tqdm import tqdm

def getEdge(emb, w, nsim):
    """
        WAP v u w for all v in emb for a given w
    """
    unionlist = list()
    for v in tqdm(list(emb.keys())[1700:1800]):
        vec = np.logical_or(emb[v], emb[w])
        word = str()
        sim = -1000
        for x in emb:
            if 1 - spatial.distance.cosine(vec, emb[x]) > sim and x != '<TOP>' and x != '<BOT>' and x != v and x != w:
                word = x
                sim = 1 - spatial.distance.cosine(vec, emb[x])
        unionlist.append((v, w, word))
    return unionlist


if __name__ == "__main__":
    dirname = '../MODELS/'
    fname = 'wiki-news-300d-1M.bin'
    NDIMS = 300
    VOCAB = 50000
    nsim = 10
    word_vecs = preprocess.load_embedding(os.path.join(dirname, fname), VOCAB)
    word_vecs = preprocess.normalize(word_vecs, 0, NDIMS)
    word_vecs = preprocess.discretize(word_vecs, 0, NDIMS)
    
    w = 'crimson'
    for i in getEdge(word_vecs, w, nsim):
        print(i)
