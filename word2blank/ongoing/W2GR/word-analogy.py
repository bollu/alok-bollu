import argparse
import numpy as np
import sys

def get_emb(mat_file):
    f = open(mat_file, 'r', errors='ignore')
    tmp = f.readlines()
    dimension = [int(x) for x in tmp[0].split(' ')]
    contents = tmp[1:]
    word_emb = np.zeros((dimension[0], dimension[1]*dimension[1]))
    vocabulary = {}
    vocabulary_inv = {}
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = np.array([float(ele) for ele in vec])
        mat = np.reshape(vec,(dimension[2], dimension[1]))
        sym_mat = mat.T@mat
        word_emb[i] = np.reshape(sym_mat, -1)
        vocabulary[word] = i
        vocabulary_inv[i] = word

    return word_emb, vocabulary, vocabulary_inv


def distance(W, vocab, ivocab, input_term):
    vecs = {}
    if len(input_term.split(' ')) < 3:
        print("Only %i words were entered.. three words are needed at the input to perform the calculation\n" % len(input_term.split(' ')))
        return 
    else:
        for idx, term in enumerate(input_term.split(' ')):
            if term in vocab:
                print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
                vecs[idx] = W[vocab[term], :]
            else:
                print('Word: %s  Out of dictionary!\n' % term)
                return

        vec_result = vecs[1] - vecs[0] + vecs[2]
        
        vec_norm = np.zeros(vec_result.shape)
        d = (np.sum(vec_result ** 2,) ** (0.5))
        vec_norm = (vec_result.T / d).T

        dist = np.dot(W, vec_norm.T)

        for term in input_term.split(' '):
            index = vocab[term]
            dist[index] = -np.Inf

        a = np.argsort(-dist)[:N]

        print("\n                               Word       Projection Norm\n")
        print("---------------------------------------------------------\n")
        for x in a:
            print("%35s\t\t%f\n" % (ivocab[x], dist[x]))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--emb_file', default='jose.txt')
    args = parser.parse_args()
    print(args)
    N = 40;          # number of closest words that will be shown
    W, vocab, ivocab = get_emb(args.emb_file)
    while True:
        input_term = input("\nEnter three words (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            distance(W, vocab, ivocab, input_term)
