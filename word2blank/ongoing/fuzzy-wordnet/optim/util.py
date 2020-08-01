from gensim.models.keyedvectors import KeyedVectors
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import networkx as nx
import re

def load_embedding(fpath, VOCAB):
    emb = dict()
    try:
        wv_from_bin = KeyedVectors.load_word2vec_format(fpath, limit=VOCAB, binary=True)
    except EOFError:
        # fucking fasttext fuck you
        wv_from_bin = KeyedVectors.load_word2vec_format(fpath, limit=VOCAB, binary=False)
    for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
        coefs = np.asarray(vector, dtype='float32')
        # if not re.match(r'\w+', word):
        if not re.match(r'[a-zA-Z]+', word):
            continue
        elif word.lower() not in emb:
            emb[word.lower()] = coefs
        else:
            emb[word.lower()] = np.mean([emb[word.lower()], coefs], axis=0)
    return emb

class wordMatrix:	# VOCAB x NDIMS matrix containing row-wise word embeddings 

	def build(emb):
		tempWordList = list(emb.keys())
		ind_keys = {i:tempWordList[i] for i in range(len(tempWordList))}
		word_keys = {tempWordList[i]:i for i in range(len(tempWordList))}
		word_mat = np.array(list(emb.values()))
		return ind_keys, word_keys, word_mat

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

	def build(sim_mat,seed=0.6,mode='absolute'):

		print("bulding adj matrix (mode:",end=' ')

		if mode not in ['absolute','mean','indegree']:
			mode = 'absolute'

		np.fill_diagonal(sim_mat, 0)	# ignore self-loops

		if mode is 'absolute':	# thresh = seed
			print('absolute)')
			for i in range(np.shape(sim_mat)[0]):
				thresh = seed
				sub_threshold_indices = sim_mat[i] < thresh
				sim_mat[i][sub_threshold_indices] = 0

		elif mode is 'mean':	# thresh = mean(row) + seed
			print('mean)')
			for i in range(np.shape(sim_mat)[0]):
				thresh = np.mean(sim_mat[i]) + seed
				sub_threshold_indices = sim_mat[i] < thresh
				sim_mat[i][sub_threshold_indices] = 0

		elif mode is 'indegree':	# n = seed // no thresh //
			print('indegree)')
			for i in range(np.shape(sim_mat)[0]):
				simRow = [[sim_mat[i][j], j] for j in range(np.shape(sim_mat)[1])]
				simRow.sort(reverse=True)
				candidate_indices = [word[1] for word in simRow[:int(seed)]]
				sim_mat[i][[j for j in range(np.shape(sim_mat)[1]) if j not in candidate_indices]] = 0

		adj_mat = sim_mat
		return adj_mat

class graphInfo:

	def decluster(comp, parent, tree, sim_mat, thresh, rate):
		# print("comp\n",comp,"parent=",parent,"thresh=",thresh)
		node_index = parent

		alt_mat = sim_mat[list(comp),:][:,list(comp)] # filtering mat for rows of words in comp only
		mapping = dict()
		i=0
		for word in comp:
			mapping[i]=word
			i+=1
		# print(mapping)
		adj_mat = adjMatrix.build(alt_mat,seed=thresh,mode='absolute')
		g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.DiGraph) 
		g = nx.relabel_nodes(g,mapping)
		scc = nx.strongly_connected_components(g)

		scc = list(scc)
		# print(scc)

		comp_len = 0
		for comp in scc:
			comp_len+=1
		if comp_len is 1:
			# print('Redundant Iteration (No new node created)')
			return graphInfo.decluster(comp, parent, tree, sim_mat, thresh+rate, rate)
				# try another iter with a higher threshold
		else:
			for compC in scc:	# child component of parent node
				if len(compC) is 1:
					for word in compC:
						# print("new edge W:",parent,"->",word)
						tree.add_edge(parent,word)
				else: 
					# print("new node added F:", node_index+1)
					new_node = node_index = node_index+1	# new null node created
					tree.add_edge(parent,new_node)
					# print("new edge N:",parent,"->",new_node)
					node_index, tree = graphInfo.decluster(compC, new_node, tree, sim_mat, thresh+rate, rate)
			return node_index, tree

	def singleton_analysis(sim_mat, init_thresh=0.5, rate=0.1):

		tree = nx.DiGraph()
		node_index = len(sim_mat)-1
		# print("node_index=",node_index)

		adj_mat = adjMatrix.build(sim_mat,init_thresh)
		g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.DiGraph) 
		scc = nx.strongly_connected_components(g)
		scc = list(scc)
		# print("SCC")
		# for comp in scc:
		# 	print(comp)

		root = node_index = node_index+1
		# print("root=",root)
		for comp in scc:
			if len(comp) is 1:
				# print(comp)
				for word in comp:
					# print("comp len = 1,",word)
					tree.add_edge(root,word)
			else:
				# print("new node added:", node_index+1)
				new_node = node_index = node_index+1	# new null node created
				tree.add_edge(root,new_node)
				node_index, tree = graphInfo.decluster(comp, new_node, tree, sim_mat, init_thresh+rate, rate)

		return tree

	def custom_tree(comp, sim_mat, init_thresh, rate):
		tree = nx.DiGraph()
		if len(comp) is 1:
			print("Specify atleast 2 tokens")

		node_index = len(sim_mat)	
		node_index, tree = graphInfo.decluster(comp, node_index, tree, sim_mat, init_thresh, rate)
		return tree
