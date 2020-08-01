import util
from util import wordMatrix as wm, adjMatrix as am, graphInfo as gi
from os import path
from tqdm import tqdm
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import write_dot, to_pydot
import sys
import argparse

sys.setrecursionlimit(10**7) 

if __name__ == '__main__':

    print("libs loaded")
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('-e', '--embeddings', type=str, help='Path to embeddings file')
    parser.add_argument('-L', '--VOCAB', type=int, help='Limit of vocab size')
    parser.add_argument('-i', '--input', type=str, help='Input File')
    parser.add_argument('-o', '--output', type=str, help='Output graph path')
    
    args = parser.parse_args()
    # dirname = '../../../utilities/MODELS/'
    # fname = 'gensim_glove_vectors.txt'
    # fname = 'wiki-news-300d-1M.bin'
    # fname = 'GoogleNews-vectors-negative300.bin'
    fname = args.embeddings
    VOCAB = args.VOCAB
    emb = util.load_embedding(fname, VOCAB)
    ind_keys, word_keys, word_mat = wm.build(emb)
    word_mat = wm.normalize(word_mat, 0)
    word_mat = wm.discretize(word_mat, 0)
    sim_mat = wm.similarity(word_mat)
    print("similarity matrix made")

	# SINGLETON ANALYSIS
	# tree = gi.singleton_analysis(sim_mat,0.5,0.001)
	# tree = nx.relabel_nodes(tree,ind_keys)
	# dot = to_pydot(tree)
	# write_dot(tree,"tree.dot")

	# CUSTOM SINGLETON ANALYSIS
    with open(args.input, 'r') as f:
        lines = f.readlines()
    tree_words = [l.strip('\n').strip() for l in lines]
    comp = [word_keys[word] for word in tree_words if word in word_keys]
    tree = gi.custom_tree(comp, sim_mat, 0.5, 0.003)
    tree = nx.relabel_nodes(tree, ind_keys)
    write_dot(tree, args.output)

	# adj_mat = am.build(sim_mat,seed=0.00,mode='mean')
	# g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.DiGraph)

	# adjList = g.adj
	# for word in adjList:
	# 	print(ind_keys[word],':',[ind_keys[i] for i in adjList[word]])

	#1 SCC
	# scc = list(nx.strongly_connected_components(g))
	# print("No. of components:",len(scc))
	# key_list = list(word_keys.keys())
	# val_list = list(word_keys.values())
	# for comp in scc:
	# 	for word in comp:
	# 		print(key_list[val_list.index(word)], end=" ")
	# 	print("")

	# #2 Planarity
	# print(nx.check_planarity(g))

	# #3 Centrality
	# print("BC")
	# bc = nx.betweenness_centrality(g)
	# dc = nx.degree_centrality(g)
	# sorted_keys = list(word_keys.keys())
	# sorted_keys.sort()
	# for word in sorted_keys:
	# 	ind = word_keys[word]
	# 	print(ind_keys[ind],bc[ind],dc[ind])
	# 	print("{: >20} {: >20} {: >20}".format(*[ind_keys[word],bc[word],dc[word]]))

	#4 Is_Connected
	# wcc = list(nx.weakly_connected_components(g))
	# print('\n',[[ind_keys[w] for w in comp] for comp in wcc])
	# print("Is connected:", len(wcc) is 1)

	#5 Print Graph
	# sub_g = nx.relabel_nodes(g,ind_keys)
	# sub_g = nx.subgraph(sub_g,['good','better','best','worst','poor','hot','cold','warm','man','woman','men','women','he','she','education','school','university','angry','fast','blue','green','on','over','below','through','around','never','always','before'])
	# write_dot(sub_g,'graph.dot')

	# #6 Similarity Histogram
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

