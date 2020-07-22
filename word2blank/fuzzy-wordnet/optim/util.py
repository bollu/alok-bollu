from gensim.models.keyedvectors import KeyedVectors
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import numpy as np
import re

def load_embedding(fpath, VOCAB):
    emb = dict()
    wv_from_bin = KeyedVectors.load_word2vec_format(fpath, limit=VOCAB)
    for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
        coefs = np.asarray(vector, dtype='float32')
        if not re.match(r'\w+', word):
            continue
        elif word.lower() not in emb:
            emb[word.lower()] = coefs
        else:
            emb[word.lower()] = np.mean([emb[word.lower()], coefs], axis=0)
    return emb

class wordMatrix:	# VOCAB x NDIMS matrix containing row-wise word embeddings 

	def build(emb):
		tempWordList = list(emb.keys())
		word_keys = {tempWordList[i]:i for i in range(len(tempWordList))}
		word_mat = np.array(list(emb.values()))
		return word_keys, word_mat

	def normalize(word_mat, axis):
		word_mat = np.exp(word_mat)			# e^x for x in all vectors
		word_mat = preprocessing.normalize(word_mat, norm='l1', axis=axis)
		return word_mat

	def discretize(word_mat, axis):	# axis: 0:column-wise ; 1:row-wise
	    threshold = np.mean(word_mat, axis=axis)
	    word_mat = (word_mat >= threshold) * 1
	    return word_mat

	def similarity(word_mat):
		sim_mat = cosine_similarity(word_mat)
		return sim_mat

	def plotHist(sim_mat, wordList, word_keys):
		res = 100
		bins = [i/res for i in range(res+1) ]
		sim_row = np.zeros(np.shape(sim_mat)[1])
		for word in wordList:
			sim_row += sim_mat[word_keys[word]]
		sim_row /= len(wordList)
		plt.hist(sim_row,bins)
		plt.show()

class adjMatrix:

	def set_threshold(simRow):
		# return np.mean(simRow) + 0.15
		return 0

	def build(sim_mat):
		np.fill_diagonal(sim_mat, 0)
		for i in range(np.shape(sim_mat)[0]):
			thresh = adjMatrix.set_threshold(sim_mat[i])
			sub_threshold_indices = sim_mat[i] < thresh
			sim_mat[i][sub_threshold_indices] = 0
		adj_mat = sim_mat
		return adj_mat
