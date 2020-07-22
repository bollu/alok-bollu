import util
from util import wordMatrix as wm, adjMatrix as am
from os import path
from tqdm import tqdm
import numpy as np
import networkx as nx

if __name__ == '__main__':

	print("libs loaded")

	dirname = '~/GitHubRepos'
	fname = 'wiki-news-300d-1M.vec'
	VOCAB = 10000
	emb = util.load_embedding(path.join(dirname, fname), VOCAB)
	word_keys, word_mat = wm.build(emb)
	word_mat = wm.normalize(word_mat, 0)
	word_mat = wm.discretize(word_mat, 0)
	sim_mat = wm.similarity(word_mat)

	adj_mat = am.build(sim_mat)
	g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.DiGraph)

	# #1 SCC
	# for comp in nx.strongly_connected_components(g):
	# 	key_list = list(word_keys.keys())
	# 	val_list = list(word_keys.values())
	# 	for word in comp:
	# 		print(key_list[val_list.index(word)], end=" ")
	# 	print("")

	#2 Planarity
	# print(nx.check_planarity(g))

	# #3 Betweenness Centrality
	# print(nx.betweenness_centrality(g))

	# #Similarity Histogram
	# # wordList = ['at','for','that','it','to']
	# wordList = emb.keys()
	# vec = np.zeros(300)
	# for word in wordList:
	# 	vec+=word_mat[word_keys[word]]
	# vec/=len(wordList)
	# print('Mean:',np.mean(vec))
	# print('Var:',np.var(vec))
	# print('Std:',np.std(vec))
	# wm.plotHist(word_mat, wordList, word_keys)
