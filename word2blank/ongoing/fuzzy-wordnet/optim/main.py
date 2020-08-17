import util
from util import wordMatrix as wm, adjMatrix as am, graphInfo as gi
from os import path, listdir
from tqdm import tqdm
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from networkx.drawing.nx_pydot import write_dot, to_pydot
from scipy.spatial.distance import cosine
import sys
import pickle as pkl

sys.setrecursionlimit(10**7) 

if __name__ == '__main__':

    print("libs loaded")

    dirname = '~/GitHubRepos'
    fname = 'wiki-news-300d-1M.vec'
    VOCAB = 5000
    emb = util.load_embedding(path.join(dirname, fname), VOCAB)
    # emb = util.load_embedding('/home/kvaditya/GitHubRepos/glove.6B.300d.txt', VOCAB,typ='glove')
    print("embeddings loaded")
    ind_keys, word_keys, word_mat = wm.build(emb)
    word_mat = wm.normalize(word_mat, 0)
    word_mat = wm.discretize(word_mat, 0)
    sim_mat = wm.similarity(word_mat)
    # sim_mat = wm.xor_similarity(word_mat)
    # print("similarity matrix made")
    # print(sim_mat)


    # quit = False
    # while(not quit):
    #     try:
    #         inp = input("Enter spaced out args a,b,c from a:b::c:? \n").split(' ')
    #         analogyList = list()
    #         w_a, w_b, w_c = inp[0], inp[1], inp[2]
    #         indA, indB, indC = word_keys[w_a], word_keys[w_b], word_keys[w_c] 
    #     except:
    #         print("retry")
    #         continue
    #     util.check_analogy(w_a,w_b,w_c,word_mat,word_keys,ind_keys)
    #     quit = 'n'==input('continue? (y/n) ')

    # total, correct = 0, 0
    # mypath = '/home/kvaditya/GitHubRepos/alok-bollu/word2blank/utilities/glove/eval/question-data'
    # files = [f for f in listdir(mypath) if path.isfile(path.join(mypath, f))]
    # for file in files:
    #     Ftotal, Fcorrect = 0, 0
    #     with open(path.join(mypath,file)) as infile:
    #         for line in tqdm(infile):
    #             try:
    #                 wA,wB,wC,wD = line.rstrip('\n').split(' ')
    #                 iA,iB,iC,iD = [word_keys[w] for w in [wA,wB,wC,wD]]
    #                 Ftotal += 1  # valid example
    #             except:
    #                 continue
    #             wE = util.check_analogy(wA,wB,wC,word_mat,word_keys,ind_keys)
    #             Fcorrect += int(wE==wD)
    #     Ftotal += 0.0001
    #     print(file.split('.')[0],'C:',Fcorrect,'T:',Ftotal,'A:',Fcorrect/Ftotal)
    #     correct+=Fcorrect
    #     total+=Ftotal
    # print('TOTAL:','C:',correct,'T:',total,'A:',correct/total)

    # SINGLETON ANALYSIS
    # temp_mat = sim_mat.copy()
    # # temp_mat = wm.similarity(sim_mat)
    # tree = gi.singleton_analysis(temp_mat,0.2,0.01)
    # tree = nx.relabel_nodes(tree,ind_keys)
    # dot = to_pydot(tree)
    # write_dot(tree,"tree.dot")

    # CUSTOM SINGLETON ANALYSIS
    # tree_words = ['go','going','gone','went','pull','pulled','be','am','is','was','will','would','could','should','what','where','why','who','when','how','here','there','then','now','that','this','never','always','ever','sometimes']
    # # tree_words = ['good','better','best','worst','poor','hot','cold','warm','man','woman','men','women','he','she','it','education','school','bridge','river','bank','money','may','might','prpbably','march','april','marched']
    # # tree_words = ['king','queen','man','woman','boy','girl']
    # comp = [word_keys[word] for word in tree_words if word in word_keys]
    # tree = gi.custom_tree(comp, sim_mat, 0.2, 0.001)
    # tree = nx.relabel_nodes(tree,ind_keys)
    # write_dot(tree,"tree.dot")

    # Hyperlex
    # quit = False
    # while(not quit):
    #     try:
    #         focus, e, d, c = input("Enter focus, edgeThresh, degThresh, clusterThresh\n").split(" ")
    #     except:
    #         print("Retry!")
    #         continue
    #     # default values for VOCAB=5000: <word> 1.2 20 0.4
    #     # edgeThresh between 1.15 and 1.23
    #     tree = gi.hyperlex(focus,sim_mat,word_keys,ind_keys,edgeThresh=float(e),degThresh=int(d),clusterThresh=float(c))
    #     tree = nx.relabel_nodes(tree,ind_keys)
    #     print(tree.adj)
    #     dot = to_pydot(tree)
    #     write_dot(tree,"tree.dot")
    #     quit = 'n'==input("continue? (y/n)")

    # temp_mat = sim_mat.copy()
    # adj_mat = am.build(temp_mat,seed=1.3,mode='mean')
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

