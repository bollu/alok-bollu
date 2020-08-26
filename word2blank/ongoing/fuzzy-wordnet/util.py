from gensim.models.keyedvectors import KeyedVectors
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as nm
from scipy.spatial.distance import cosine
from matplotlib import pyplot as plt
from itertools import combinations
import numpy as np
from numpy import logical_and as AND, logical_or as OR, logical_not as NOT, logical_xor as XOR
from tqdm import tqdm
import networkx as nx
import re
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from pyeda.inter import *

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

    def rank(sim_mat,limit=1):
        cutoff = int(limit*sim_mat.shape[0])
        rank_mat = np.zeros(sim_mat.shape)
        for u in range(sim_mat.shape[0]):
            vec = sim_mat[u]
            simList = [[vec[i],i] for i in range(vec.shape[0])]
            simList.sort(reverse=True)
            for r in range(min(len(simList),cutoff)):   # r = rank
                rank_mat[u][simList[r][1]] = r
        return rank_mat

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

    def build(sim_mat,seed=0.6,mode='absolute',reverse=False):

        print("bulding adj matrix (mode:",end=' ')

        if mode not in ['absolute','mean','indegree']:
            mode = 'absolute'

        np.fill_diagonal(sim_mat, 0)	# ignore self-loops

        if mode is 'absolute':	# thresh = seed
            print('absolute)')
            for i in range(np.shape(sim_mat)[0]):
                thresh = seed
                if reverse:
                    sub_threshold_indices = sim_mat[i] > thresh
                else:
                    sub_threshold_indices = sim_mat[i] < thresh
                sim_mat[i][sub_threshold_indices] = 0

        elif mode is 'mean':	# thresh = mean(row) + seed
            print('mean)')
            for i in range(np.shape(sim_mat)[0]):
                thresh = seed*np.mean(sim_mat[i])
                if reverse:
                    sub_threshold_indices = sim_mat[i] > thresh
                else:
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

D = 300
def OR(*argv):
    res = np.zeros(D,dtype=int)
    for arg in argv:
        res = res|arg
    return res

def AND(*argv):
    res = np.ones(D,dtype=int)
    for arg in argv:
        res = res&arg
    return res


def check_analogy(wA,wB,wC,word_mat,word_keys,ind_keys,boolInd):
    def sim(x,y):
        return np.sum(np.multiply(x,y))/(np.linalg.norm(x)*np.linalg.norm(y))
    def xor_sim(x,y):
        return (300-np.sum(np.logical_xor(x,y)))/300

    indA, indB, indC = word_keys[wA], word_keys[wB], word_keys[wC] 
    vecA, vecB, vecC = word_mat[indA,:], word_mat[indB,:], word_mat[indC,:] 
    # vecA, vecB, vecC = nm(word_mat[indA,:],norm='l1'), nm(word_mat[indB,:],norm='l1'), nm(word_mat[indC,:],norm='l1') 
    vecD = vecB - vecA + vecC
    # vecD = np.logical_xor(vecC,np.logical_xor(vecA,vecB))
    # A,B,C = vecA.astype('bool'), vecB.astype('bool'), vecC.astype('bool')
    # vecD = OR(OR(AND(NOT(A),NOT(C)),AND(B,C)),AND(B,NOT(A)))
    # vecD = OR(AND(NOT(C),OR(AND(A,NOT(B)),AND(NOT(A),B))),AND(C,OR(AND(NOT(A),NOT(B)),AND(A,B))))
    # vecD = OR(OR(AND(A,AND(NOT(B),NOT(C))),AND(NOT(A),C)),OR(AND(NOT(A),B),AND(B,C)))
    # vecD = OR(AND(B,NOT(XOR(A,C))),AND(NOT(A),AND(NOT(B),C)))
    # vecD = minBoolForm([A,B,C],boolInd)
    # print(vecD)
    simScore = np.zeros(len(ind_keys))
    for indE in ind_keys:
        if indE in [indA,indB,indC]:
            simScore[indE] = 0.0
            continue
        vecE = word_mat[indE,:]
        simScore[indE] = sim(vecD,vecE)
    simCand = simScore.argsort()[-1:][::-1]
    simList = [[ind_keys[ind],simScore[ind]] for ind in simCand]
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

def minBoolForm(x,i):
    if i == 0 :
        return np.zeros(D)
    elif i == 1 :
        return (AND(x[0], x[1], x[2]),)
    elif i == 2 :
        return (AND(~x[0], x[1], x[2]),)
    elif i == 3 :
        return (AND(x[1], x[2]),)
    elif i == 4 :
        return (AND(x[0], ~x[1], x[2]),)
    elif i == 5 :
        return (AND(x[0], x[2]),)
    elif i == 6 :
        return (OR(AND(~x[0], x[1], x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 7 :
        return (OR(AND(x[1], x[2]), AND(x[0], x[2])),)
    elif i == 8 :
        return (AND(~x[0], ~x[1], x[2]),)
    elif i == 9 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(x[0], x[1], x[2])),)
    elif i == 10 :
        return (AND(~x[0], x[2]),)
    elif i == 11 :
        return (OR(AND(~x[0], x[2]), AND(x[1], x[2])),)
    elif i == 12 :
        return (AND(~x[1], x[2]),)
    elif i == 13 :
        return (OR(AND(~x[1], x[2]), AND(x[0], x[2])),)
    elif i == 14 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], x[2])),)
    elif i == 15 :
        return (x[2],)
    elif i == 16 :
        return (AND(x[0], x[1], ~x[2]),)
    elif i == 17 :
        return (AND(x[0], x[1]),)
    elif i == 18 :
        return (OR(AND(x[0], x[1], ~x[2]), AND(~x[0], x[1], x[2])),)
    elif i == 19 :
        return (OR(AND(x[1], x[2]), AND(x[0], x[1])),)
    elif i == 20 :
        return (OR(AND(x[0], x[1], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 21 :
        return (OR(AND(x[0], x[1]), AND(x[0], x[2])),)
    elif i == 22 :
        return (OR(AND(x[0], x[1], ~x[2]), AND(~x[0], x[1], x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 23 :
        return (OR(AND(x[1], x[2]), AND(x[0], x[1]), AND(x[0], x[2])),)
    elif i == 24 :
        return (OR(AND(x[0], x[1], ~x[2]), AND(~x[0], ~x[1], x[2])),)
    elif i == 25 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(x[0], x[1])),)
    elif i == 26 :
        return (OR(AND(~x[0], x[2]), AND(x[0], x[1], ~x[2])),)
    elif i == 27 :
        return (OR(AND(~x[0], x[2]), AND(x[0], x[1])),)
    elif i == 28 :
        return (OR(AND(~x[1], x[2]), AND(x[0], x[1], ~x[2])),)
    elif i == 29 :
        return (OR(AND(~x[1], x[2]), AND(x[0], x[1])),)
    elif i == 30 :
        return (OR(AND(~x[1], x[2]), AND(x[0], x[1], ~x[2]), AND(~x[0], x[2])),)
    elif i == 31 :
        return (OR(x[2], AND(x[0], x[1])),)
    elif i == 32 :
        return (AND(~x[0], x[1], ~x[2]),)
    elif i == 33 :
        return (OR(AND(~x[0], x[1], ~x[2]), AND(x[0], x[1], x[2])),)
    elif i == 34 :
        return (AND(~x[0], x[1]),)
    elif i == 35 :
        return (OR(AND(~x[0], x[1]), AND(x[1], x[2])),)
    elif i == 36 :
        return (OR(AND(~x[0], x[1], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 37 :
        return (OR(AND(~x[0], x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 38 :
        return (OR(AND(~x[0], x[1]), AND(x[0], ~x[1], x[2])),)
    elif i == 39 :
        return (OR(AND(~x[0], x[1]), AND(x[0], x[2])),)
    elif i == 40 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(~x[0], x[1], ~x[2])),)
    elif i == 41 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(~x[0], x[1], ~x[2]), AND(x[0], x[1], x[2])),)
    elif i == 42 :
        return (OR(AND(~x[0], x[2]), AND(~x[0], x[1])),)
    elif i == 43 :
        return (OR(AND(~x[0], x[2]), AND(~x[0], x[1]), AND(x[1], x[2])),)
    elif i == 44 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], x[1], ~x[2])),)
    elif i == 45 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 46 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], x[1])),)
    elif i == 47 :
        return (OR(x[2], AND(~x[0], x[1])),)
    elif i == 48 :
        return (AND(x[1], ~x[2]),)
    elif i == 49 :
        return (OR(AND(x[1], ~x[2]), AND(x[0], x[1])),)
    elif i == 50 :
        return (OR(AND(x[1], ~x[2]), AND(~x[0], x[1])),)
    elif i == 51 :
        return (x[1],)
    elif i == 52 :
        return (OR(AND(x[1], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 53 :
        return (OR(AND(x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 54 :
        return (OR(AND(x[1], ~x[2]), AND(~x[0], x[1]), AND(x[0], ~x[1], x[2])),)
    elif i == 55 :
        return (OR(x[1], AND(x[0], x[2])),)
    elif i == 56 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(x[1], ~x[2])),)
    elif i == 57 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(x[0], x[1]), AND(x[1], ~x[2])),)
    elif i == 58 :
        return (OR(AND(~x[0], x[2]), AND(x[1], ~x[2])),)
    elif i == 59 :
        return (OR(x[1], AND(~x[0], x[2])),)
    elif i == 60 :
        return (OR(AND(~x[1], x[2]), AND(x[1], ~x[2])),)
    elif i == 61 :
        return (OR(AND(~x[1], x[2]), AND(x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 62 :
        return (OR(AND(~x[0], x[2]), AND(x[1], ~x[2]), AND(~x[1], x[2])),)
    elif i == 63 :
        return (OR(x[1], x[2]),)
    elif i == 64 :
        return (AND(x[0], ~x[1], ~x[2]),)
    elif i == 65 :
        return (OR(AND(x[0], x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 66 :
        return (OR(AND(~x[0], x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 67 :
        return (OR(AND(x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 68 :
        return (AND(x[0], ~x[1]),)
    elif i == 69 :
        return (OR(AND(x[0], ~x[1]), AND(x[0], x[2])),)
    elif i == 70 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[1], x[2])),)
    elif i == 71 :
        return (OR(AND(x[0], ~x[1]), AND(x[1], x[2])),)
    elif i == 72 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 73 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(x[0], x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 74 :
        return (OR(AND(~x[0], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 75 :
        return (OR(AND(~x[0], x[2]), AND(x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 76 :
        return (OR(AND(x[0], ~x[1]), AND(~x[1], x[2])),)
    elif i == 77 :
        return (OR(AND(x[0], ~x[1]), AND(x[0], x[2]), AND(~x[1], x[2])),)
    elif i == 78 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[2])),)
    elif i == 79 :
        return (OR(x[2], AND(x[0], ~x[1])),)
    elif i == 80 :
        return (AND(x[0], ~x[2]),)
    elif i == 81 :
        return (OR(AND(x[0], ~x[2]), AND(x[0], x[1])),)
    elif i == 82 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], x[1], x[2])),)
    elif i == 83 :
        return (OR(AND(x[0], ~x[2]), AND(x[1], x[2])),)
    elif i == 84 :
        return (OR(AND(x[0], ~x[2]), AND(x[0], ~x[1])),)
    elif i == 85 :
        return (x[0],)
    elif i == 86 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], x[1], x[2]), AND(x[0], ~x[1])),)
    elif i == 87 :
        return (OR(x[0], AND(x[1], x[2])),)
    elif i == 88 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], ~x[1], x[2])),)
    elif i == 89 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], ~x[1], x[2]), AND(x[0], x[1])),)
    elif i == 90 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], x[2])),)
    elif i == 91 :
        return (OR(AND(x[0], ~x[2]), AND(x[1], x[2]), AND(~x[0], x[2])),)
    elif i == 92 :
        return (OR(AND(x[0], ~x[2]), AND(~x[1], x[2])),)
    elif i == 93 :
        return (OR(x[0], AND(~x[1], x[2])),)
    elif i == 94 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], x[2]), AND(x[0], ~x[2])),)
    elif i == 95 :
        return (OR(x[0], x[2]),)
    elif i == 96 :
        return (OR(AND(~x[0], x[1], ~x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 97 :
        return (OR(AND(~x[0], x[1], ~x[2]), AND(x[0], x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 98 :
        return (OR(AND(~x[0], x[1]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 99 :
        return (OR(AND(~x[0], x[1]), AND(x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 100 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[1], ~x[2])),)
    elif i == 101 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 102 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[1])),)
    elif i == 103 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[1]), AND(x[1], x[2])),)
    elif i == 104 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(~x[0], x[1], ~x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 105 :
        return (OR(AND(~x[0], ~x[1], x[2]), AND(~x[0], x[1], ~x[2]), AND(x[0], x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 106 :
        return (OR(AND(~x[0], x[2]), AND(~x[0], x[1]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 107 :
        return (OR(AND(~x[0], x[2]), AND(~x[0], x[1]), AND(x[1], x[2]), AND(x[0], ~x[1], ~x[2])),)
    elif i == 108 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[1], ~x[2]), AND(~x[1], x[2])),)
    elif i == 109 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[1], ~x[2]), AND(x[0], x[2]), AND(~x[1], x[2])),)
    elif i == 110 :
        return (OR(AND(~x[1], x[2]), AND(x[0], ~x[1]), AND(~x[0], x[1])),)
    elif i == 111 :
        return (OR(x[2], AND(x[0], ~x[1]), AND(~x[0], x[1])),)
    elif i == 112 :
        return (OR(AND(x[0], ~x[2]), AND(x[1], ~x[2])),)
    elif i == 113 :
        return (OR(AND(x[0], ~x[2]), AND(x[1], ~x[2]), AND(x[0], x[1])),)
    elif i == 114 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], x[1])),)
    elif i == 115 :
        return (OR(x[1], AND(x[0], ~x[2])),)
    elif i == 116 :
        return (OR(AND(x[0], ~x[1]), AND(x[1], ~x[2])),)
    elif i == 117 :
        return (OR(x[0], AND(x[1], ~x[2])),)
    elif i == 118 :
        return (OR(AND(x[0], ~x[1]), AND(x[1], ~x[2]), AND(~x[0], x[1])),)
    elif i == 119 :
        return (OR(x[0], x[1]),)
    elif i == 120 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], ~x[1], x[2]), AND(x[1], ~x[2])),)
    elif i == 121 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], ~x[1], x[2]), AND(x[0], x[1]), AND(x[1], ~x[2])),)
    elif i == 122 :
        return (OR(AND(x[0], ~x[2]), AND(x[1], ~x[2]), AND(~x[0], x[2])),)
    elif i == 123 :
        return (OR(x[1], AND(x[0], ~x[2]), AND(~x[0], x[2])),)
    elif i == 124 :
        return (OR(AND(x[0], ~x[2]), AND(x[1], ~x[2]), AND(~x[1], x[2])),)
    elif i == 125 :
        return (OR(x[0], AND(~x[1], x[2]), AND(x[1], ~x[2])),)
    elif i == 126 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], x[1]), AND(~x[1], x[2])),)
    elif i == 127 :
        return (OR(x[0], x[1], x[2]),)
    elif i == 128 :
        return (AND(~x[0], ~x[1], ~x[2]),)
    elif i == 129 :
        return (OR(AND(~x[0], ~x[1], ~x[2]), AND(x[0], x[1], x[2])),)
    elif i == 130 :
        return (OR(AND(~x[0], x[1], x[2]), AND(~x[0], ~x[1], ~x[2])),)
    elif i == 131 :
        return (OR(AND(x[1], x[2]), AND(~x[0], ~x[1], ~x[2])),)
    elif i == 132 :
        return (OR(AND(~x[0], ~x[1], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 133 :
        return (OR(AND(~x[0], ~x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 134 :
        return (OR(AND(~x[0], x[1], x[2]), AND(~x[0], ~x[1], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 135 :
        return (OR(AND(x[1], x[2]), AND(~x[0], ~x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 136 :
        return (AND(~x[0], ~x[1]),)
    elif i == 137 :
        return (OR(AND(~x[0], ~x[1]), AND(x[0], x[1], x[2])),)
    elif i == 138 :
        return (OR(AND(~x[0], x[2]), AND(~x[0], ~x[1])),)
    elif i == 139 :
        return (OR(AND(~x[0], ~x[1]), AND(x[1], x[2])),)
    elif i == 140 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], ~x[1])),)
    elif i == 141 :
        return (OR(AND(~x[0], ~x[1]), AND(x[0], x[2])),)
    elif i == 142 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], ~x[1]), AND(~x[0], x[2])),)
    elif i == 143 :
        return (OR(x[2], AND(~x[0], ~x[1])),)
    elif i == 144 :
        return (OR(AND(x[0], x[1], ~x[2]), AND(~x[0], ~x[1], ~x[2])),)
    elif i == 145 :
        return (OR(AND(~x[0], ~x[1], ~x[2]), AND(x[0], x[1])),)
    elif i == 146 :
        return (OR(AND(x[0], x[1], ~x[2]), AND(~x[0], ~x[1], ~x[2]), AND(~x[0], x[1], x[2])),)
    elif i == 147 :
        return (OR(AND(x[1], x[2]), AND(~x[0], ~x[1], ~x[2]), AND(x[0], x[1])),)
    elif i == 148 :
        return (OR(AND(x[0], x[1], ~x[2]), AND(~x[0], ~x[1], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 149 :
        return (OR(AND(~x[0], ~x[1], ~x[2]), AND(x[0], x[1]), AND(x[0], x[2])),)
    elif i == 150 :
        return (OR(AND(x[0], x[1], ~x[2]), AND(~x[0], ~x[1], ~x[2]), AND(~x[0], x[1], x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 151 :
        return (OR(AND(x[1], x[2]), AND(~x[0], ~x[1], ~x[2]), AND(x[0], x[1]), AND(x[0], x[2])),)
    elif i == 152 :
        return (OR(AND(x[0], x[1], ~x[2]), AND(~x[0], ~x[1])),)
    elif i == 153 :
        return (OR(AND(~x[0], ~x[1]), AND(x[0], x[1])),)
    elif i == 154 :
        return (OR(AND(~x[0], x[2]), AND(x[0], x[1], ~x[2]), AND(~x[0], ~x[1])),)
    elif i == 155 :
        return (OR(AND(~x[0], ~x[1]), AND(x[1], x[2]), AND(x[0], x[1])),)
    elif i == 156 :
        return (OR(AND(~x[1], x[2]), AND(x[0], x[1], ~x[2]), AND(~x[0], ~x[1])),)
    elif i == 157 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], ~x[1]), AND(x[0], x[1])),)
    elif i == 158 :
        return (OR(AND(~x[1], x[2]), AND(x[0], x[1], ~x[2]), AND(~x[0], ~x[1]), AND(~x[0], x[2])),)
    elif i == 159 :
        return (OR(x[2], AND(~x[0], ~x[1]), AND(x[0], x[1])),)
    elif i == 160 :
        return (AND(~x[0], ~x[2]),)
    elif i == 161 :
        return (OR(AND(x[0], x[1], x[2]), AND(~x[0], ~x[2])),)
    elif i == 162 :
        return (OR(AND(~x[0], x[1]), AND(~x[0], ~x[2])),)
    elif i == 163 :
        return (OR(AND(x[1], x[2]), AND(~x[0], ~x[2])),)
    elif i == 164 :
        return (OR(AND(~x[0], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 165 :
        return (OR(AND(~x[0], ~x[2]), AND(x[0], x[2])),)
    elif i == 166 :
        return (OR(AND(~x[0], x[1]), AND(~x[0], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 167 :
        return (OR(AND(x[1], x[2]), AND(~x[0], ~x[2]), AND(x[0], x[2])),)
    elif i == 168 :
        return (OR(AND(~x[0], ~x[1]), AND(~x[0], ~x[2])),)
    elif i == 169 :
        return (OR(AND(~x[0], ~x[1]), AND(x[0], x[1], x[2]), AND(~x[0], ~x[2])),)
    elif i == 170 :
        return (~x[0],)
    elif i == 171 :
        return (OR(~x[0], AND(x[1], x[2])),)
    elif i == 172 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], ~x[2])),)
    elif i == 173 :
        return (OR(AND(~x[1], x[2]), AND(~x[0], ~x[2]), AND(x[0], x[2])),)
    elif i == 174 :
        return (OR(~x[0], AND(~x[1], x[2])),)
    elif i == 175 :
        return (OR(~x[0], x[2]),)
    elif i == 176 :
        return (OR(AND(x[1], ~x[2]), AND(~x[0], ~x[2])),)
    elif i == 177 :
        return (OR(AND(x[0], x[1]), AND(~x[0], ~x[2])),)
    elif i == 178 :
        return (OR(AND(x[1], ~x[2]), AND(~x[0], ~x[2]), AND(~x[0], x[1])),)
    elif i == 179 :
        return (OR(x[1], AND(~x[0], ~x[2])),)
    elif i == 180 :
        return (OR(AND(x[1], ~x[2]), AND(~x[0], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 181 :
        return (OR(AND(x[1], ~x[2]), AND(~x[0], ~x[2]), AND(x[0], x[2])),)
    elif i == 182 :
        return (OR(AND(x[1], ~x[2]), AND(~x[0], x[1]), AND(~x[0], ~x[2]), AND(x[0], ~x[1], x[2])),)
    elif i == 183 :
        return (OR(x[1], AND(~x[0], ~x[2]), AND(x[0], x[2])),)
    elif i == 184 :
        return (OR(AND(~x[0], ~x[1]), AND(x[1], ~x[2])),)
    elif i == 185 :
        return (OR(AND(~x[0], ~x[1]), AND(x[1], ~x[2]), AND(x[0], x[1])),)
    elif i == 186 :
        return (OR(~x[0], AND(x[1], ~x[2])),)
    elif i == 187 :
        return (OR(~x[0], x[1]),)
    elif i == 188 :
        return (OR(AND(~x[1], x[2]), AND(x[1], ~x[2]), AND(~x[0], ~x[2])),)
    elif i == 189 :
        return (OR(AND(~x[1], x[2]), AND(x[0], x[1]), AND(~x[0], ~x[2])),)
    elif i == 190 :
        return (OR(~x[0], AND(~x[1], x[2]), AND(x[1], ~x[2])),)
    elif i == 191 :
        return (OR(~x[0], x[1], x[2]),)
    elif i == 192 :
        return (AND(~x[1], ~x[2]),)
    elif i == 193 :
        return (OR(AND(x[0], x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 194 :
        return (OR(AND(~x[0], x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 195 :
        return (OR(AND(x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 196 :
        return (OR(AND(x[0], ~x[1]), AND(~x[1], ~x[2])),)
    elif i == 197 :
        return (OR(AND(~x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 198 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 199 :
        return (OR(AND(~x[1], ~x[2]), AND(x[1], x[2]), AND(x[0], x[2])),)
    elif i == 200 :
        return (OR(AND(~x[0], ~x[1]), AND(~x[1], ~x[2])),)
    elif i == 201 :
        return (OR(AND(~x[0], ~x[1]), AND(x[0], x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 202 :
        return (OR(AND(~x[0], x[2]), AND(~x[1], ~x[2])),)
    elif i == 203 :
        return (OR(AND(~x[0], x[2]), AND(x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 204 :
        return (~x[1],)
    elif i == 205 :
        return (OR(~x[1], AND(x[0], x[2])),)
    elif i == 206 :
        return (OR(~x[1], AND(~x[0], x[2])),)
    elif i == 207 :
        return (OR(~x[1], x[2]),)
    elif i == 208 :
        return (OR(AND(x[0], ~x[2]), AND(~x[1], ~x[2])),)
    elif i == 209 :
        return (OR(AND(x[0], x[1]), AND(~x[1], ~x[2])),)
    elif i == 210 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 211 :
        return (OR(AND(x[0], ~x[2]), AND(x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 212 :
        return (OR(AND(x[0], ~x[2]), AND(x[0], ~x[1]), AND(~x[1], ~x[2])),)
    elif i == 213 :
        return (OR(x[0], AND(~x[1], ~x[2])),)
    elif i == 214 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], x[1], x[2]), AND(x[0], ~x[1]), AND(~x[1], ~x[2])),)
    elif i == 215 :
        return (OR(x[0], AND(x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 216 :
        return (OR(AND(x[0], ~x[2]), AND(~x[0], ~x[1])),)
    elif i == 217 :
        return (OR(AND(~x[0], ~x[1]), AND(x[0], x[1]), AND(~x[1], ~x[2])),)
    elif i == 218 :
        return (OR(AND(x[0], ~x[2]), AND(~x[1], ~x[2]), AND(~x[0], x[2])),)
    elif i == 219 :
        return (OR(AND(~x[0], x[2]), AND(x[0], x[1]), AND(~x[1], ~x[2])),)
    elif i == 220 :
        return (OR(~x[1], AND(x[0], ~x[2])),)
    elif i == 221 :
        return (OR(x[0], ~x[1]),)
    elif i == 222 :
        return (OR(~x[1], AND(x[0], ~x[2]), AND(~x[0], x[2])),)
    elif i == 223 :
        return (OR(x[0], ~x[1], x[2]),)
    elif i == 224 :
        return (OR(AND(~x[0], ~x[2]), AND(~x[1], ~x[2])),)
    elif i == 225 :
        return (OR(AND(~x[0], ~x[2]), AND(x[0], x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 226 :
        return (OR(AND(~x[0], x[1]), AND(~x[1], ~x[2])),)
    elif i == 227 :
        return (OR(AND(~x[1], ~x[2]), AND(x[1], x[2]), AND(~x[0], ~x[2])),)
    elif i == 228 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], ~x[2])),)
    elif i == 229 :
        return (OR(AND(~x[0], ~x[2]), AND(~x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 230 :
        return (OR(AND(x[0], ~x[1]), AND(~x[0], x[1]), AND(~x[1], ~x[2])),)
    elif i == 231 :
        return (OR(AND(~x[0], x[1]), AND(~x[1], ~x[2]), AND(x[0], x[2])),)
    elif i == 232 :
        return (OR(AND(~x[0], ~x[2]), AND(~x[0], ~x[1]), AND(~x[1], ~x[2])),)
    elif i == 233 :
        return (OR(AND(~x[0], ~x[2]), AND(~x[0], ~x[1]), AND(x[0], x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 234 :
        return (OR(~x[0], AND(~x[1], ~x[2])),)
    elif i == 235 :
        return (OR(~x[0], AND(x[1], x[2]), AND(~x[1], ~x[2])),)
    elif i == 236 :
        return (OR(~x[1], AND(~x[0], ~x[2])),)
    elif i == 237 :
        return (OR(~x[1], AND(~x[0], ~x[2]), AND(x[0], x[2])),)
    elif i == 238 :
        return (OR(~x[0], ~x[1]),)
    elif i == 239 :
        return (OR(~x[0], ~x[1], x[2]),)
    elif i == 240 :
        return (~x[2],)
    elif i == 241 :
        return (OR(~x[2], AND(x[0], x[1])),)
    elif i == 242 :
        return (OR(~x[2], AND(~x[0], x[1])),)
    elif i == 243 :
        return (OR(x[1], ~x[2]),)
    elif i == 244 :
        return (OR(~x[2], AND(x[0], ~x[1])),)
    elif i == 245 :
        return (OR(x[0], ~x[2]),)
    elif i == 246 :
        return (OR(~x[2], AND(x[0], ~x[1]), AND(~x[0], x[1])),)
    elif i == 247 :
        return (OR(x[0], x[1], ~x[2]),)
    elif i == 248 :
        return (OR(~x[2], AND(~x[0], ~x[1])),)
    elif i == 249 :
        return (OR(~x[2], AND(~x[0], ~x[1]), AND(x[0], x[1])),)
    elif i == 250 :
        return (OR(~x[0], ~x[2]),)
    elif i == 251 :
        return (OR(~x[0], x[1], ~x[2]),)
    elif i == 252 :
        return (OR(~x[1], ~x[2]),)
    elif i == 253 :
        return (OR(x[0], ~x[1], ~x[2]),)
    elif i == 254 :
        return (OR(~x[0], ~x[1], ~x[2]),)
    elif i == 255 :
        return np.ones(D)