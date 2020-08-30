import numpy as np
import codecs
from numpy.linalg import norm
from scipy.stats import spearmanr
from os import path
from tqdm import tqdm

def preproc_vectors(fpath):
    tot, dim = 0, 0
    line_number = 0
    word_vectors = dict()
    with open(fpath,'r') as infile:
        for line in infile:
            # print(line)
            if line_number is 0:
                tot, dim = line.rstrip('\n').split(' ')
            else:
                line = line.rstrip('\n').split(' ')
                word, vec = line[0], np.asarray(line[1:], dtype='float')
                word_vectors[word] = vec
            line_number += 1
    print('dim =',dim,'total =',tot)
    return word_vectors

def poincare_distance(u,v):
    return np.arccosh(1+2*(norm(u-v)**2)/((1-norm(u)**2)*(1-norm(v)**2)))

# as mentioned in the poincare paper
# score(is-a(u, v)) = −(1 + α(|v| − |u|))d(u, v); α (penalty if u>v) default = 1000  
def custom_score(u,v,alpha=1000):
    t = -(1+alpha*(norm(v)-norm(u)))*poincare_distance(u,v)
    print(t)
    return t

def hyperlex_analysis(word_vectors, source="hyperlex"):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    """
    pair_list = []
    if source == "hyperlex":
        fread_simlex=codecs.open("hyperlex.txt", 'r', 'utf-8')
    elif source == "hyperlex-nouns":
        fread_simlex=codecs.open("hyperlex-nouns.txt", 'r', 'utf-8')
    elif source == "hyperlex-test":
        fread_simlex=codecs.open("hyperlex_test.txt", 'r', 'utf-8')
    elif source == "hyperlex-train":
        fread_simlex=codecs.open("hyperlex_train.txt", 'r', 'utf-8')
    else:
        "Error with HyperLex!"

    print('reading hyperlex files')

    line_number = 0
    for line in fread_simlex:

        if line_number > 0:

            tokens = line.split()
            word_i = tokens[0].lower()
            word_j = tokens[1].lower()
            score = float(tokens[2])

            if word_i in word_vectors and word_j in word_vectors:
                pair_list.append( ((word_i, word_j), score) )
            else:
                pass

        line_number += 1

    if not pair_list:
        return (0.0, 0)

    pair_list.sort(key=lambda x: - x[1])

    coverage = len(pair_list)

    extracted_list = []
    extracted_scores = {}

    print('calculating scores')

    for (x,y) in tqdm(pair_list):

        (word_i, word_j) = x
        current_distance = custom_score(word_vectors[word_i], word_vectors[word_j])   
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)
    return round(spearman_rho[0], 3), coverage

def main():
    dirname = 'models'
    # fname = 'wiki-news-300d-nouns.txt'
    fname = 'trmsa_nouns.vec'
    fpath = path.join(dirname, fname)
    print('begun')
    word_vectors = preproc_vectors(fpath)
    print('word vectors processed')
    print('hyperlex_analysis begun')
    rho, coverage = hyperlex_analysis(word_vectors,source='hyperlex')
    print('analysis done')
    print('rho =',rho,'coverage =',coverage)

if __name__=='__main__':
    main()
