import os
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing

def load_bin_vec(fname):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in tqdm(range(vocab_size)):
            word = []
            while True:
               ch = f.read(1)
               if ch == b' ':
                   word = ''.join([str(i).strip('b\'').strip('\'') for i in word])
                   break
               if ch != b'\n':
                   word.append(ch)
            word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')
    return word_vecs

def normalize(word_vecs, l_dim):
    """
        l_dim(0): sum up the columns
        l_dim(1): sum up the rows
    """
    vec_mat = np.array(list(word_vecs.values()))
    vec_mat = np.exp(vec_mat)                       # e^x for x in all vectors
    if l_dim == 0:
        vec_mat = preprocessing.normalize(vec_mat, norm='l1', axis=0)
    elif l_dim == 1:
        vec_mat = preprocessing.normalize(vec_mat, norm='l1', axis=1)
    
    vecs = np.vsplit(vec_mat, len(word_vecs.keys()))
    word_vecs = zip(list(word_vecs.keys()), vecs)
    return dict(word_vecs)
        

def discretize(word_vecs, threshold):
    """
        forall x in word_vecs[word]::
            x = 1 if x > threshold
            x = 0 otherwise
    """
    pass

def construct_ontology(word_vecs, desc):
    """
        desc(0): discretized
        desc(1): not discretized
    """
    pass

if __name__ == '__main__':
    dirname = '../MODELS/'
    fname = 'vanilla200.bin'
    word_vecs = load_bin_vec(os.path.join(dirname, fname))
    word_vecs = normalize(word_vecs, 0)
    print(np.reshape(word_vecs['the'], 200))

