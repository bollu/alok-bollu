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

def plot(mat, fname):
    X, Y, U, V= zip(*mat)
    plt.figure()
    ax = plt.gca()
    ax.quiver(X, Y, U, V, angles='uv', scale_units='xy', scale=0.1, edgecolor=['red', 'blue', 'green', 'yellow'])
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    plt.draw()
    plt.show()
    plt.savefig(fname)

if __name__ == '__main__':
    emb1 = load_embedding('posvel_models/symp-size4-text8.bin', True)
    emb2 = load_embedding('posvel_models/syn1neg-symp-size4-text8.bin', True)
    wordlist = ['king', 'man', 'queen', 'woman']
    mat1 = get_embs(emb1, wordlist)
    mat2 = get_embs(emb2, wordlist)
    print(mat1)
    print(mat2)
    plot(mat1, 'rays.png')
    plot(mat2, 'syn1neg-rays.png')
