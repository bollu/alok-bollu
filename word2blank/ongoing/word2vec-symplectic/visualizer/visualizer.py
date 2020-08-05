import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
import gensim
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import sys
from sklearn import decomposition

def load_embedding(fpath, x, wordlist, dim):
    print("Loading embeddings...")
    emb = dict()
    wv_from_bin = KeyedVectors.load_word2vec_format(fpath, binary=x)
    embs = np.array([wv_from_bin.get_vector(w) for w in wv_from_bin.vocab])
    vpos, vmom  = np.hsplit(embs, 2)
    pca = decomposition.PCA(n_components=dim)
    pca.fit(vpos)
    pca_posvec = pca.transform(vpos)
    pca.fit(vmom)
    pca_momvec = pca.transform(vmom)
    vs = np.concatenate((pca_posvec, pca_momvec), axis=1)
    wvs = []
    for word in wordlist:
        ix = wv_from_bin.vocab[word].index
        wvs.append(vs[ix])
    wvs = np.array(wvs)
    return wvs


def plot(mat, fname, dim):
    X = 0
    Y = 0
    Z = 0
    U = 0
    V = 0
    W = 0
    fig = 0
    ax = 0
    if dim == 3:
        X, Y, Z, U, V, W = zip(*mat)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, Z, U, V, W)
        ax.set_zlim([-1, 1])
    elif dim == 2:
        X, Y, U, V = zip(*mat)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.quiver(X, Y, U, V, units='xy', angles='uv')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.draw()
    plt.show()
    plt.savefig(fname)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: %s <path to embeddings file>" % (sys.argv[1], ))
    wordlist = ['france', 'paris', 'germany', 'berlin']
    dim = 2
    mat1 = load_embedding(sys.argv[1], True, wordlist, dim)
    print(mat1)
    plot(mat1, 'rays.png', dim)
