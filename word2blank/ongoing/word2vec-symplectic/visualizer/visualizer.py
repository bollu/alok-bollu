import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import gensim
from tqdm import tqdm

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

def plot(mat):
    U, V, X, Y = zip(*mat)
    plt.figure()
    ax = plt.gca()
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=0.1)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    plt.draw()
    plt.show()
    plt.savefig('rays.png')

if __name__ == '__main__':
    emb = load_embedding('posvel_models/symp-size4-text8.bin', True)
    wordlist = ['good', 'better', 'best', 'bad', 'worse', 'worst']
    mat = get_embs(emb, wordlist)
    print(mat)
    plot(mat)
