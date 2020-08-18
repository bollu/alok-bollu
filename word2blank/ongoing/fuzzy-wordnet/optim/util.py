from gensim.models.keyedvectors import KeyedVectors
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as nm
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt
from itertools import combinations
import numpy as np
from numpy import logical_and as AND, logical_or as OR, logical_not as NOT
from tqdm import tqdm
import networkx as nx
import re
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def load_embedding(fpath, VOCAB, typ='w2v'):
    emb = dict()
    if typ is 'glove':
        glove_file = datapath(fpath)
        tmp_file = get_tmpfile("test_word2vec.txt")
        _ = glove2word2vec(glove_file, tmp_file)
        wv_from_bin = KeyedVectors.load_word2vec_format(tmp_file, limit=VOCAB)
    elif typ is 'w2v':
        wv_from_bin = KeyedVectors.load_word2vec_format(fpath, limit=VOCAB)
        
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
        word_mat = preprocessing.normalize(word_mat, norm='l2', axis=axis)
        return word_mat

    def discretize(word_mat, axis):	# axis: 0:column-wise ; 1:row-wise
        threshold = np.mean(word_mat, axis=axis)
        word_mat = (word_mat >= threshold) * 1
        return word_mat

    def similarity(word_mat):
        sim_mat = cosine_similarity(word_mat)
        return sim_mat

    def xor_similarity(word_mat):
        m = word_mat.shape[0]
        sim_mat = np.zeros((m,m))
        for i in tqdm(range(m)):
            for j in range(m):
                    val = (300-np.sum(np.logical_xor(word_mat[i],word_mat[j])))/300
                    # print(val)
                    sim_mat[i][j] = val
                    # print(sim_mat[i][j])

        return sim_mat

    # def reform(sim_mat):
    # 	rMax, rMin = np.full(np.shape(sim_mat),0.99), np.min(sim_mat, axis=1)	# row wise MAX & MIN
    # 	sim_mat = (sim_mat.transpose()-rMin.transpose()).transpose()
    # 	return np.divide(sim_mat,rMax)

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
                thresh = seed*np.mean(sim_mat[i])
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
        if not comp:
            comp = range(np.shape(sim_mat)[0])
        print(comp)
        tree = nx.DiGraph()
        if len(comp) is 1:
            print("Specify atleast 2 tokens")

        node_index = len(sim_mat)	
        node_index, tree = graphInfo.decluster(comp, node_index, tree, sim_mat, init_thresh, rate)
        return tree

    def hyperlex(focus,sim_mat,word_keys,ind_keys,edgeThresh=1.2,degThresh=4,clusterThresh=0.9):

        focus = word_keys[focus]
        mean = np.mean(sim_mat,axis=1)	# for each row // vector
        thresh = edgeThresh*np.mean(sim_mat[focus,:])
        print("mean & thresh calced")
        vertices = [ind for ind in range(np.shape(sim_mat)[0]) if ind > 100 and sim_mat[focus][ind] >= thresh and ind!=focus]	# words important to 'focus'
        H,D = list(), dict()
        for ind in vertices:
            inwardList = list(np.where(sim_mat[:,ind]>=edgeThresh*mean)[0])
            D[ind]=[len(inwardList),inwardList]
        print("Degree calced")
        def clusterCoeff(ind):
            inDeg, inwardList = D[ind]
            if(len(inwardList)==0):
                return 0
            c, crossEdges = 0, list(combinations(inwardList,2))
            for i,j in crossEdges:
                c += sim_mat[i][j]
            return c/len(crossEdges)

        def goodCandidate(ind):
            return D[ind][0] > degThresh and clusterCoeff(ind) > clusterThresh

        S = {ind:sim_mat[focus][ind] for ind in D.keys()}

        while S:
            print("len S:",len(S))
            # vertices sorted by decreasing degree
            V = [ind for ind, val in sorted(S.items(), key=lambda x: x[1], reverse=True)]
            if goodCandidate(V[0]):
                print("New candidate:",ind_keys[V[0]])
                H.append(V[0])
                for node in D[V[0]][1]:
                    if node in S: del S[node]
            if V[0] in S: del S[V[0]]

        adj_mat = np.zeros(np.shape(sim_mat))
        for i in vertices:
            for j in vertices:
                if sim_mat[i][j] > edgeThresh*mean[i]:
                    adj_mat[i][j] = 1 - sim_mat[i][j]

        # ignore self-loops	
        np.fill_diagonal(adj_mat, 0)
        
        # MUST BE INCL IN MST
        for ind in H:
            adj_mat[focus][ind] = 0.1

        g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.Graph)	
        for i in range(np.shape(adj_mat)[0]):
            if(i not in vertices and i != focus):
                g.remove_node(i)
        print("focus graph made")
        tree = nx.minimum_spanning_tree(g)
        print(tree)
        print("mst made")
        return tree 
        

def check_analogy(wA,wB,wC,word_mat,word_keys,ind_keys):
    def sim(x,y):
        return np.sum(np.multiply(x,y))/(np.linalg.norm(x)*np.linalg.norm(y))
    indA, indB, indC = word_keys[wA], word_keys[wB], word_keys[wC] 
    vecA, vecB, vecC = word_mat[indA,:], word_mat[indB,:], word_mat[indC,:] 
    # vecA, vecB, vecC = nm(word_mat[indA,:],norm='l1'), nm(word_mat[indB,:],norm='l1'), nm(word_mat[indC,:],norm='l1') 
    vecD = vecB - vecA + vecC
    # vecD = np.logical_xor(vecC,np.logical_xor(vecA,vecB))
    A,B,C = vecA, vecB, vecC
    # vecD = OR(OR(AND(NOT(A),NOT(C)),AND(B,C)),AND(B,NOT(A)))
    # vecD = OR(AND(NOT(C),OR(AND(A,NOT(B)),AND(NOT(A),B))),AND(C,OR(AND(NOT(A),NOT(B)),AND(A,B))))
    simScore = np.zeros(len(ind_keys))
    for indE in ind_keys:
        if indE in [indA,indB,indC]:
            simScore[indE] = 0.0
            continue
        vecE = word_mat[indE,:]
        simScore[indE] = sim(vecD,vecE)
        # simScore[indE] = (1 - cosine(np.logical_and(vecA,vecB),np.logical_and(vecC,vecE)))
    # simCand = simScore.argsort()[-5:][::-1] # Top 5 words
    simCand = simScore.argsort()[-1:][::-1]
    simList = [[ind_keys[ind],simScore[ind]] for ind in simCand]
    # print(simList)
    return simList[0][0]

def dimensionalSimilarity(word_mat,word_keys,ind_keys,mode='xor'):

    m, dim = np.shape(word_mat)

    if mode is 'xor':
        xor_mat = np.zeros((m,m,dim))
        for i in range(m):
            for j in range(m):
                xor_mat[i][j] = np.logical_xor(word_mat[i],word_mat[j])
        return xor_mat

    elif mode is 'xnor':
        xnor_mat = np.zeros(m,m,dim)
        for i in range(m):
            for j in range(m):
                xnor_mat[i][j] = np.logical_not(np.logical_xor(word_mat[i],word_mat[j]))
        return xnor_mat