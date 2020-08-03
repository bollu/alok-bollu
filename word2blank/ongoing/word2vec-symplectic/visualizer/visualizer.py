import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import gensim
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import sys

def load_embedding(fpath, x):
    print("Loading embeddings...")
    emb = dict()
    wv_from_bin = KeyedVectors.load_word2vec_format(fpath, binary=x, limit=1000000)
    for word, vector in tqdm(zip(wv_from_bin.vocab, wv_from_bin.vectors)):
        coefs = np.asarray(vector, dtype='float32')
        if word not in emb:
            emb[word] = coefs
    return emb

def get_embs(emb, wordlist):
    mat = list()
    for word in wordlist:
        mat.append(emb[word])
    mat = np.array(mat)
    return mat

def plot(mat, fname):
    X, Y, Z, U, V, W = zip(*mat)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    plt.draw()
    plt.show()
    plt.savefig(fname)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: %s <path to embeddings file>" % (sys.argv[1], ))
    emb1 = load_embedding(sys.argv[1], True)
    # emb2 = load_embedding('posvel_models/syn1neg-symp-size4-dim6-text8.bin', True)
    wordlist = ['good', 'better', 'best', 'bad', 'worse', 'worst']
    mat1 = get_embs(emb1, wordlist)
    # mat2 = get_embs(emb2, wordlist)
    print(mat1)
    # print(mat2)
    plot(mat1, 'rays.png')
    # plot(mat2, 'syn1neg-rays.png')
