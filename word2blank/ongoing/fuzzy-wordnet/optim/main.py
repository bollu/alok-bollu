import util
from util import wordMatrix as wm, adjMatrix as am, graphInfo as gi
from os import path
from tqdm import tqdm
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from networkx.drawing.nx_pydot import write_dot, to_pydot
import sys

# sys.setrecursionlimit(10**7) 

if __name__ == '__main__':

	print("libs loaded")

	dirname = '~/GitHubRepos'
	fname = 'wiki-news-300d-1M.vec'
	VOCAB = 5000
	emb = util.load_embedding(path.join(dirname, fname), VOCAB)
	ind_keys, word_keys, word_mat = wm.build(emb)
	word_mat = wm.normalize(word_mat, 0)
	word_mat = wm.discretize(word_mat, 0)
	sim_mat = wm.similarity(word_mat)
	# sim_mat = wm.reform(sim_mat) 

	# king = sim_mat[word_keys['king']]
	# queen = sim_mat[word_keys['queen']]
	# man = sim_mat[word_keys['man']]
	# woman = sim_mat[word_keys['woman']]

	print("similarity matrix made")

	# SINGLETON ANALYSIS
	# temp_mat = sim_mat.copy()
	# # temp_mat = wm.similarity(sim_mat)
	# tree = gi.singleton_analysis(temp_mat,0.5,0.001)
	# tree = nx.relabel_nodes(tree,ind_keys)
	# dot = to_pydot(tree)
	# write_dot(tree,"tree.dot")

	# CUSTOM SINGLETON ANALYSIS
	# tree_words = ['good','better','best','worst','poor','hot','cold','warm','man','woman','men','women','he','she','it']
	# tree_words = ['king','queen','man','woman','boy','girl']
	# comp = [word_keys[word] for word in tree_words if word in word_keys]
	# tree = gi.custom_tree(comp, sim_mat, 0.2, 0.001)
	# tree = nx.relabel_nodes(tree,ind_keys)
	# write_dot(tree,"tree.dot")

	# Hyperlex
	quit = False
	while(not quit):
		try:
			focus, e, d, c = input("Enter focus, edgeThresh, degThresh, clusterThresh\n").split(" ")
		except:
			print("Retry!")
			continue
		tree = gi.hyperlex(focus,sim_mat,word_keys,ind_keys,edgeThresh=float(e),degThresh=int(d),clusterThresh=float(c))
		tree = nx.relabel_nodes(tree,ind_keys)
		print(tree.adj)
		dot = to_pydot(tree)
		write_dot(tree,"tree.dot")
		quit = 'n'==input("continue? (y/n)")

	# temp_mat = sim_mat.copy()
	# # temp_mat = wm.similarity(sim_mat)
	# adj_mat = am.build(temp_mat,seed=1.1,mode='mean')
	# g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.DiGraph)
	# print("Graph made")

	# tree = nx.relabel_nodes(tree,ind_keys)
	# dot = to_pydot(tree)
	# write_dot(tree,"tree.dot")

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

	#3 Centrality
	# print("Betweenness Centrality:")
	# bc = nx.betweenness_centrality(g)
	# bc_sorted = [[bc[ind],ind_keys[ind]] for ind in bc]
	# bc_sorted.sort(reverse=True)
	# for word in bc_sorted:
	# 	print("{: >20} {: >20}".format(*[word[1],word[0]]))

	#4 Is_Connected
	# wcc = list(nx.weakly_connected_components(g))
	# print('\n',[[ind_keys[w] for w in comp] for comp in wcc])
	# print("Is connected:", len(wcc) is 1)

	#5 Analogy
	# a:b::c:d
	# quit = False
	# while(not quit):
	# 	inp = input("Enter spaced out args a,b,c from a:b::c:? \n").split(' ')
	# 	analogyList = list()
	# 	w_a, w_b, w_c = inp[0], inp[1], inp[2]
	# 	# w_a, w_b, w_c = 'india', 'delhi', 'china'
	# 	valid = True
	# 	for w in inp:
	# 		if w not in word_keys:
	# 			print(w,'not in vocab')
	# 			valid = False
	# 	if not valid:
	# 		continue
	# 	a, b, c = word_keys[w_a], word_keys[w_b], word_keys[w_c]

		# p_ac = nx.shortest_path(g, source=a, target=c)
		# p_bc = nx.shortest_path(g, source=b, target=c)
		# print([ind_keys[ind] for ind in p_ac])
		# print([ind_keys[ind] for ind in p_bc])
		# junction = None
		# for i in range(min(len(p_ac), len(p_bc))):
		# 	pos = -1-i
		# 	junction = i
		# 	if(p_ac[pos]!=p_bc[pos]):
		# 		break
		# divList_a, divList_b = p_ac[:-junction], p_bc[:-junction] 
		# print([ind_keys[ind] for ind in divList_a])
		# print([ind_keys[ind] for ind in divList_b])

		# candidateList = dict(g.adj[b]).keys()
		# for ind in candidateList:
		# 	if sim_mat[b][ind] > sim_mat[a][ind] and sim_mat[c][ind] > sim_mat[a][ind]:
		# 		analogyList.append([sim_mat[c][ind],ind_keys[ind]])
		# analogyList.sort(reverse=True)
		# print(analogyList) 
		# quit = ('n'==input("Do you wish to continue? (y/n) "))

	#5.5
	# quit = False
	# while(not quit):
	# 	a,b,c = input("Enter spaced A:B::C\n").split(" ")
	# 	util.check_analogy(a,b,c,sim_mat,word_keys,ind_keys)
	# 	quit = ('n'==input('Proceed? (y/n) '))

	# # 6 Print Graph
	# sub_g = nx.relabel_nodes(g,ind_keys)
	# sub_g = nx.subgraph(sub_g,['king','queen','man','woman','boy','girl'])
	# write_dot(sub_g,'graph.dot')

	# #7 Similarity Histogram
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

