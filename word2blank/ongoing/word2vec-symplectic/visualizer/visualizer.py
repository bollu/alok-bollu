import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.decomposition import PCA
import gensim
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import sys

def load_embedding(fpath, x, wordlist):
    print("Loading embeddings...")
    emb = dict()
    wv_from_bin = KeyedVectors.load_word2vec_format(fpath, binary=x)
    embs = np.array([wv_from_bin.get_vector(w) for w in wordlist])
    vpos, vmom  = np.hsplit(embs, 2)
    vpos = np.corrcoef(vpos)
    vmom = np.corrcoef(vmom)
    posval, posvec = np.linalg.eig(vpos)
    momval, momvec = np.linalg.eig(vmom)
    ixs = np.argsort(-posval)
    posval = posvec[ixs]
    posvec = posvec[:, ixs]
    pca_posvec = posvec[:, :3]
    ixs = np.argsort(-momval)
    momval = momvec[ixs]
    momvec = momvec[:, ixs]
    pca_momvec = momvec[:, :3]
    vs = np.concatenate((pca_posvec, pca_momvec), axis=1)
    return vs


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
    # emb1 = load_embedding(sys.argv[1], True)
    # emb2 = load_embedding('posvel_models/syn1neg-symp-size4-dim6-text8.bin', True)
    wordlist = ['good', 'better', 'best', 'bad', 'worse', 'worst']
    mat1 = load_embedding(sys.argv[1], True, wordlist)
    # mat2 = get_embs(emb2, wordlist)
    print(mat1)
    # print(mat2)
    plot(mat1, 'rays.png')
    # plot(mat2, 'syn1neg-rays.png')
