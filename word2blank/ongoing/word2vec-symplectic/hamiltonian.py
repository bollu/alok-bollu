from gensim.models import KeyedVectors
import numpy as np
import random

def load_vectors(fname):
    """
        Load vectors from file, split them into `p` and `q`
    """
    print("Loading embeddings...")
    wv = KeyedVectors.load_word2vec_format(fname, binary=True)
    vocab = list(wv.vocab)
    mat = np.array([vec for vec in wv.vectors])
    p_mat, q_mat = np.hsplit(mat, 2)
    return wv, vocab, p_mat, q_mat

def shifting(p, q, dt):
    """
        p_mat[w] = wp
        q_mat[w] = wq;
        => d(wq)/dt = wq;
        => wp[i] += wq[i] * dt
    """
    for ix in range(len(p)):
        p[ix] += q[ix] * dt
    return p

def observations(wv, vocab, p, q, dt, num_iters):
    """
        Print the results over a given number of iterations
    """
    # words = random.sample(vocab[10:2000], k=1)
    words = ['fire']
    print("Words: " + str(words))
    sims = []
    print("Word \t\t Similar words")
    print('=' * 89)
    for i in range(num_iters):
        del_p = shifting(p, q, dt)
        print("Iteration: " +str(i))
        print("=" * 89)
        for w in words:
            ix = vocab.index(w)
            pix = del_p[ix]
            vec = np.concatenate((pix, q[ix]), axis=None)
            sims = wv.similar_by_vector(vec, topn=5)
            print(w + "\t\t" + str([s[0] for s in sims]))
        print('-' * 89)
    return      

if __name__ == '__main__':
    fname = './models/symp-size=400-initrandom.bin'
    wv, vocab, p, q = load_vectors(fname)
    dt = 1
    num_iters = 100
    observations(wv, vocab, p, q, dt, num_iters)
