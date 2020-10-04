import re
import sys
import csv
import subprocess
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pyeda.inter import *
from os import path, listdir
from itertools import combinations
from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import write_dot, to_pydot
from scipy.spatial.distance import cosine
import sklearn.preprocessing as preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


sys.setrecursionlimit(10**7) 

########################################################################################
                                # UTILITY FUNCTIONS #
########################################################################################

def load_embedding(fpath, VOCAB, typ='w2v'):
    '''
        description:
            - given path to KeyedVect file, returns embeddings dictionary
        params: 
            - fpath: file path to KeyedVect file
            - VOCAB: vocabulary size (final vocab size may not match this due to the cleaning)
            - typ:
                - 'w2v': std word2vec KeyedVectors file
                - 'glove': std glove .txt format
        returns:
            - emb: vocab wide embeddging dictionary
    '''
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
        # cleaning
        if not re.match(r'[a-zA-Z]+', word):
            continue
        elif word.lower() not in emb:
            emb[word.lower()] = coefs
        else:
            emb[word.lower()] = np.mean([emb[word.lower()], coefs], axis=0)
    return emb

########################################################################################

def buildWordMat(emb):
    '''
        description:
            - given embedding dict, returns embeddings matrix
        params: 
            - emb: embeddings dictionary
        returns:
            - indKeys: {i:w for ind,w in enumerate(vocab)} (index to word mapping) (dict)
            - wordKeys: {w:i for ind,w in enumerate(vocab)} (word to index mapping) (dict)
            - word_mat: embeddings matrix (VOCABxNDIMS)
    '''
    tempWordList = list(emb.keys())
    indKeys = {i:tempWordList[i] for i in range(len(tempWordList))}
    wordKeys = {tempWordList[i]:i for i in range(len(tempWordList))}
    word_mat = np.array(list(emb.values()))
    return indKeys, wordKeys, word_mat

########################################################################################

def normalize(word_mat, axis):
    '''
        description:
            - given embeddings matrix, returns normalized embeddings matrix
        params: 
            - word_mat: embeddings matrix (VOCABxNDMIS)
            - axis:
                0: normalize each dimension/feature
                1: independently normalize each words separately
        returns:
            - indKeys: {i:w for ind,w in enumerate(vocab)} (index to word mapping) (dict)
            - wordKeys: {w:i for ind,w in enumerate(vocab)} (word to index mapping) (dict)
            - word_mat: normalized embeddings matrix (VOCABxNDIMS)
    '''
    word_mat = np.exp(word_mat)         # e^x for x in all vectors
    word_mat = preprocessing.normalize(word_mat, norm='l2', axis=axis)
    return word_mat

########################################################################################

def discretize(word_mat, axis):
    '''
        description:
            - given embeddings matrix, returns discretized(0/1) embeddings matrix
        params: 
            - word_mat: embeddings matrix (VOCABxNDMIS)
            - axis:
                0: discretize each dimension/feature
                1: discretize normalize each words separately
        returns:
            - word_mat: discretized embeddings matrix (VOCABxNDIMS)
    '''
    threshold = np.mean(word_mat, axis=axis)
    word_mat = (word_mat >= threshold) * 1
    return word_mat

########################################################################################

def cos_similarity(word_mat):
    '''
        description:
            - given embeddings matrix, returns pairwise cosine similarity matrix
        params: 
            - word_mat: embeddings matrix (VOCABxNDMIS)
        returns:
            - word_mat: similarity matrix (VOCABxVOCAB)
    '''
    sim_mat = cosine_similarity(word_mat)
    return sim_mat

########################################################################################

def xor_similarity(word_mat):
    '''
        description:
            - given discretized embeddings matrix, returns pairwise XOR similarity matrix
        params: 
            - word_mat: discretized embeddings matrix (VOCABxNDMIS)
        returns:
            - word_mat: similarity matrix (VOCABxVOCAB)
    '''
    if np.sum(np.logical_not(np.logical_or(word_mat == 0,word_mat == 1))) != 0:
        print('error: word_mat not discrete')
        return None

    m,n = word_mat.shape
    sim_mat = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
                val = (n-np.sum(np.logical_xor(word_mat[i],word_mat[j])))/n
                sim_mat[i][j] = val
    return sim_mat

########################################################################################

def rankMatrix(sim_mat,limit=1):
    '''
        description:
            - given similarity matrix, returns edge matrix with pairwise similarity rank
        params: 
            - sim_mat: similarity matrix
            - limit: edge weight set to 0 for all ranks > limit
        returns:
            - rank_mat: rank matrix
    '''
    cutoff = int(limit*sim_mat.shape[0])
    rank_mat = np.zeros(sim_mat.shape)
    for u in range(sim_mat.shape[0]):
        vec = sim_mat[u]
        simList = [[vec[i],i] for i in range(vec.shape[0])]
        simList.sort(reverse=True)
        for r in range(min(len(simList),cutoff+1)):   # r = rank
            rank_mat[u][simList[r][1]] = r
    return rank_mat

########################################################################################

def flipMatrix(rank_mat,focusRank=1):
    '''
        description:
            - given similarity rank matrix, returns edge matrix with pairwise 'flipped' similarity rank
        params: 
            - sim_mat: similarity matrix
            - focusRank: edge weight set to 0 for all ranks > limit
        returns:
            - flip_mat:
                - weights(r) > focusRank set to (0)
                - weights(r) <= focusRank set to (focusRank-r+1)
                - eg: for focusRank = 3: [1,2,3,4,5] -> [3,2,1,4,5]
    '''
    flip_mat = np.zeros(rank_mat.shape)
    for u in range(rank_mat.shape[0]):
        for v in range(rank_mat.shape[0]):
            if rank_mat[u][v] <= focusRank:
                flip_mat[u][v] = focusRank - rank_mat[u][v] + 1

########################################################################################

def plotHist(sim_mat, wordList, wordKeys):
    '''
        description:
            - given similarity matrix, display cummulative histogram of similarity for words in wordList
        params: 
            - sim_mat: raw sim value used to compute SCC
            - wordList: set of target words
            - wordKeys: word to ind mapping 
        returns:
            None
    '''
    res = 100
    bins = [i/res for i in range(res+1) ]
    sim_row = np.zeros(np.shape(sim_mat)[1])
    for word in wordList:
        sim_row += sim_mat[wordKeys[word]]
    sim_row /= len(wordList)
    plt.hist(sim_row,bins)
    plt.show()

########################################################################################

def buildAdjMat(sim_mat,thresh=0.6,mode='abs',reverse=False):
    '''
        description:
            - given a VxV similarity matrix, returns adjacency weight matrix
        params: 
            - sim_mat: raw sim value used to compute SCC
            - thresh: limit value used in edge condition (see mode for more info)
                - default values for (abs=0.6), (mean=1.15), (indegree=10)
            - mode:
                - abs: edge[wA,wB] = True if similarity > thresh
                - mean: edge[wA,wB] = True if similarity > thresh*(mean similarity of row)
                - indegree: edge[wA,wB] = True if wB ∈ Top (int)'thresh' words acc to similarity
            - reverse: applies NOT of thresh conditions if True  
        returns:
            - tree: networkx graph object containing singletons
    '''

    print("bulding adj matrix (mode:",end=' ')

    if mode not in ['abs','mean','indegree']:
        mode = 'abs'

    np.fill_diagonal(sim_mat, 0)    # ignore self-loops

    if mode is 'abs':  # edge[wA,wB] = True if similarity > thresh
        print('absolute)')
        for i in range(np.shape(sim_mat)[0]):
            if reverse:
                sub_threshold_indices = sim_mat[i] > thresh
            else:
                sub_threshold_indices = sim_mat[i] < thresh
            sim_mat[i][sub_threshold_indices] = 0

    elif mode is 'mean':    # edge[wA,wB] = True if similarity > thresh*(mean similarity of row)
        print('mean)')
        for i in range(np.shape(sim_mat)[0]):
            thresh = thresh*np.mean(sim_mat[i])
            if reverse:
                sub_threshold_indices = sim_mat[i] > thresh
            else:
                sub_threshold_indices = sim_mat[i] < thresh
            sim_mat[i][sub_threshold_indices] = 0

    elif mode is 'indegree':    # True if wB ∈ Top (int)'thresh' words acc to similarity
        print('indegree)')
        for i in range(np.shape(sim_mat)[0]):
            simRow = [[sim_mat[i][j], j] for j in range(np.shape(sim_mat)[1])]
            simRow.sort(reverse=True)
            if reverse:
                candidate_indices = [word[1] for word in simRow[int(thresh):]]
            else:
                candidate_indices = [word[1] for word in simRow[:int(thresh)]]
            sim_mat[i][[j for j in range(np.shape(sim_mat)[1]) if j not in candidate_indices]] = 0

    adj_mat = sim_mat
    return adj_mat

########################################################################################

def decluster(tree,comp,parent,sim_mat,thresh=0.2,rate=0.01):
    '''
        description:
            - given a list of words, func recursively runs SCC with increasing thresh
        params: 
            - tree: common graph object
            - comp: list of ~related words under a common parent word
            - parent: parent word to comp
            - sim_mat: raw sim value used to compute SCC
            - thresh: 'abs' thresh set to build adjMat to run SCC on
            - rate: val by which thresh increments at each deeper recursive layer (ie. thresh+=rate)

        returns:
            - node_index: int value of latest node created
            - tree: networkx graph object containing recursive SCC components
    '''
    node_index = parent
    alt_mat = sim_mat[list(comp),:][:,list(comp)] # filtering mat for rows of words in comp only
    mapping = dict()
    i=0
    for word in comp:
        mapping[i]=word
        i+=1
    adj_mat = adjMatrix.build(alt_mat,seed=thresh,mode='ab')
    g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.DiGraph) 
    g = nx.relabel_nodes(g,mapping)
    scc = nx.strongly_connected_components(g)

    scc = list(scc)

    comp_len = 0
    for comp in scc:
        comp_len+=1
    if comp_len is 1:
        # print('Redundant Iteration (No new node created)')
        return decluster(tree, comp, parent, sim_mat, thresh+rate, rate)
    else:
        for compC in scc:   # child component of parent node
            if len(compC) is 1:
                for word in compC:
                    tree.add_edge(parent,word)
            else: 
                new_node = node_index = node_index+1    # new null node created
                tree.add_edge(parent,new_node)
                node_index, tree = decluster(tree, compC, new_node, sim_mat, thresh+rate, rate)
        return node_index, tree

########################################################################################

def singleton_analysis(sim_mat,init_thresh=0.5,rate=0.1):
    '''
        description:
            - given a similarity matrix, returns relational tree where each word is a leaf
        params: 
            - sim_mat: raw sim value used to compute SCC
            - init_thresh: initial thresh (see decluster() for more info)
            - rate: val by which thresh increments at each deeper recursive layer (ie. thresh+=rate)

        returns:
            - tree: networkx graph object containing singletons
    '''
    tree = nx.DiGraph()
    node_index = len(sim_mat)-1

    adj_mat = adjMatrix.build(sim_mat,init_thresh)
    g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.DiGraph) 
    scc = nx.strongly_connected_components(g)
    scc = list(scc)

    root = node_index = node_index+1
    for comp in scc:
        if len(comp) is 1:
            for word in comp:
                tree.add_edge(root,word)
        else:
            new_node = node_index = node_index+1    # new null node created
            tree.add_edge(root,new_node)
            node_index, tree = graphInfo.decluster(tree, comp, new_node, sim_mat, init_thresh+rate, rate)

    return tree

########################################################################################

def custom_singleton_analysis(comp, sim_mat,init_thresh=0.2,rate=0.001):
    '''
        description:
            - given a custom list of words, returns relational tree where each word is a leaf
        params: 
            - comp: custom list of words (must be a subset of vocab of sim_mat)
            - sim_mat: raw sim value used to compute SCC
            - init_thresh: initial thresh (see decluster() for more info)
            - rate: val by which thresh increments at each deeper recursive layer (ie. thresh+=rate)

        returns:
            - tree: networkx graph object containing singletons
    '''
    if not comp:
        comp = range(np.shape(sim_mat)[0])
    print(comp)
    tree = nx.DiGraph()
    if len(comp) is 1:
        print("Specify atleast 2 tokens")
        return None

    node_index = len(sim_mat)   
    node_index, tree = graphInfo.decluster(tree, comp, node_index, sim_mat, init_thresh, rate)
    return tree

########################################################################################

def hyperlex(focus,sim_mat,wordKeys,indKeys,edgeThresh=1.2,degThresh=4,clusterThresh=0.9):
    '''
        description:
            - for a target word, returns hyperlex tree
            - link to paper: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.6499&rep=rep1&type=pdf
        params: 
            - focus: target word acc to which tree is to be made 
            - sim_mat: raw sim value (VxV)
            - wordKeys: {w:i for ind,w in enumerate(vocab)} (word to index mapping) (dict)
            - indKeys: {i:w for ind,w in enumerate(vocab)} (index to word mapping) (dict)
            - edgeThresh: remove edges with weight < edgeThresh
            - degThresh: remove vertices with indegree < degThresh 
            - clusterThresh: remove vertices with cluster coeff < clusterThresh

        returns:
            - tree: networkx graph object containing singletons
    '''
    focus = wordKeys[focus]
    mean = np.mean(sim_mat,axis=1)  # for each row // vector
    thresh = edgeThresh*np.mean(sim_mat[focus,:])
    print("hyperlex(): mean & thresh calced")
    vertices = [ind for ind in range(np.shape(sim_mat)[0]) if ind > 100 and sim_mat[focus][ind] >= thresh and ind!=focus]   # words important to 'focus'
    H,D = list(), dict()
    for ind in vertices:
        inwardList = list(np.where(sim_mat[:,ind]>=edgeThresh*mean)[0])
        D[ind]=[len(inwardList),inwardList]
    print("hyperlex(): degree calculated")

    def clusterCoeff(ind):
        '''
            description:
                - given index to vertex in graph, returns its 'cluster coefficient' (see paper link for more info)
            params: 
                - ind: index to vertex
            returns:
                - clusterCoeff: cluster coefficient to given vertex
        '''
        inDeg, inwardList = D[ind]
        if(len(inwardList)==0):
            return 0
        c, crossEdges = 0, list(combinations(inwardList,2))
        for i,j in crossEdges:
            c += sim_mat[i][j]
        clusterCoeff = c/len(crossEdges)
        return clusterCoeff

    def goodCandidate(ind):
        '''
            description:
                - given index to vertex in graph, returns if it is a 'good candidate'
                - returns True if 
                    - degree[vertex] > degThresh
                    AND
                    - clusterCoeff[vertex] > clusterThresh
            params: 
                - ind: index to vertex
            returns:
                - (boolean) True/False
        '''
        return D[ind][0] > degThresh and clusterCoeff(ind) > clusterThresh

    S = {ind:sim_mat[focus][ind] for ind in D.keys()}

    while S:
        print("len s:",len(S))
        # vertices sorted by decreasing degree
        V = [ind for ind, val in sorted(S.items(), key=lambda x: x[1], reverse=True)]
        if goodCandidate(V[0]):
            print("new candidate:",indKeys[V[0]])
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
    
    # add focus to tree (connect focus to all hubs in H with practically zero edge wt)
    for ind in H:
        adj_mat[focus][ind] = 1e-15

    g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.Graph)  
    for i in range(np.shape(adj_mat)[0]):
        if(i not in vertices and i != focus):
            g.remove_node(i)
    print("hyperlex(): focus graph made")
    tree = nx.minimum_spanning_tree(g)
    print(tree)
    print("hyperlex(): mst made")
    return tree

########################################################################################

def scc(graph,indKeys):
    '''
        description:
            - given a directed graph, returns its strongly connected components
        params: 
            - graph: networkx DiGraph object
            - indKeys: index to word mapping (dict)
        returns:
            - scc: strongly connected components of graph in form of a nested list of component[words[]] 
    '''
    scc = list(nx.strongly_connected_components(g))
    relabelled_scc = list()
    for comp in scc:
        relabelled_scc.append([indKeys[i] for i in comp])
    return relabelled_scc

########################################################################################

def isPlanar(graph):
    '''
        description:
            - given a graph, return if it is planar
        params: 
            - graph: networkx DiGraph object
        returns:
            - boolean True if graph is planar, False otherwise
    '''
    return nx.check_planarity(graph)

def isConnected(graph):
    '''
        description:
            - given a graph, return if it is connected (weakly)
        params: 
            - graph: networkx DiGraph object
        returns:
            - boolean True if graph is planar, False otherwise
    '''
    wcc = list(nx.weakly_connected_components(graph))
    return len(wcc) == 1

########################################################################################

def centrality(graph,indKeys,centralityType='betw'):
    '''
        description:
            - given a graph, returns vertex-wise centrality
        params: 
            - graph: networkx DiGraph object
            - indKeys: index to word mapping (dict)
            - centralityType:
                - 'betw': betweenness centrality
        returns:
            - cenSorted: word-keyed dictionary sorted in descending order by centrality
    '''
    cen = None
    if centralityType is 'betw':
        cen = nx.betweenness_centrality(graph)
    cenSorted = {[cen[ind],indKeys[ind]] for ind in cen}
    cenSorted.sort(reverse=True)
    cenSorted = {w:centrality for centrality, w in cenSorted}
    return cenSorted

########################################################################################

def plotDendrogram(word_mat):
    '''
        description:
            - given embedding matrix of any kind, returns dendrogram (heirarchical clustering)
        params: 
            - word_mat: embedding matrix (VOCABxNDIMS)
        returns:
            None; displays plot
    '''
    X = word_mat
    dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"), 
                                labels=list(wordKeys.keys()),
                                above_threshold_color='C1')
    plt.title('Dendrogram')
    plt.xlabel('Embeddings')
    plt.ylabel('Distance')
    plt.show()

########################################################################################
                                # MAIN SCRIPTS #
########################################################################################

def vecFile2vecMat(embFile,vocab=100,embType='w2v',norm=False,disc=False):
    '''
        description:
            - given path to vec file, returns vec matrix with corresponding ind-word mapping objects
        params: 
            - embFile: path to vec file
            - vocab: upper limit to size of vocabulary
            - embType:
                - 'w2v': word2vec (.vec)
                - 'glove': GloVe (.txt)
            - norm: embeddings will be normalized if set True
            - disc: embeddings will be discretized if set True
        returns:
            - word_mat: embeddings matrix (VOCABxNDIMS)
            - indKeys: index to word mapping (for embedding matrix) (dict)
            - wordKeys: word to index mapping (for embedding matrix) (dict)
    '''
    if not path.exists(embFile):
        print('error: path does not exist')
        return None
    if not path.isfile(embFile):
        print('error: path not a file')
        return None
    else:
        emb = load_embedding(embFile,vocab,typ=embType)
        indKeys, wordKeys, word_mat = buildWordMat(emb)
        if norm:
            word_mat = normalize(word_mat, axis=0)
        if disc:
            word_mat = discretize(word_mat, axis=0)
        return word_mat, indKeys, wordKeys

########################################################################################

def vecMat2adjMat(word_mat,simType='cos',edgeType='abs',edgeThresh='0.6'):
    '''
        description:
            - given path to vec file, simType, and edgeType, returns vec matrix with corresponding ind-word mapping objects
        params: 
            - word_mat: embeddings matrix (VOCABxNDIMS)
            - simType:
                - 'cos': cosine similarity
                - 'xor': xor similarity (embeddings must be discrete)
            - edgeType:
                - 'abs': edge[wA,wB] = True if similarity > thresh
                - 'mean': edge[wA,wB] = True if similarity > thresh*(mean similarity of row)
                - 'indegree': edge[wA,wB] = True if wB ∈ Top (int)'thresh' words acc to similarity
            - edgeThesh: edge threshold (thresh)
        returns:
            - adjMat: adjacency matrix (VOCABxVOCAB)
    '''
    sim_mat = None
    if simType is 'cos':
        sim_mat = cos_similarity(word_mat)
    elif simType is 'xor':
        sim_mat = xor_similarity(word_mat)
    adjMat = buildAdjMat(sim_mat,thresh=0.6,mode='abs',reverse=False)
    return adjMat

########################################################################################

def adjMat2graph(adjMat):
    '''
        description:
            - given adjacency matrix, builds and returns networkx graph object
        params: 
            - adjMat: adjacency matrix (numpy 2d-array)
        returns:
            - graph: networkx graph object
    '''
    graph = nx.convert_matrix.from_numpy_array(adjMat, create_using=nx.DiGraph)
    return graph

########################################################################################

def vecMat2tree(word_mat,indKeys,treeType='TRmsa',treeThresh=20,simType='cos',focusRank=3):
    '''
        description:
            - given embeddings matrix, treeType and simType, builds and returns networkx tree object
        params: 
            - word_mat: embeddings matrix (VOCABxNDIMS)
            - indKeys: index to word mapping (dict)
            - treeType:
                - 'msa': maximum spanning arboroscence
                - 'Rmsa': rank minimum spanning arboroscence
                - 'TRmsa': transpose rank minimum spanning arboroscence
                - 'FRmsa': flipped rank minimum spanning arboroscence (see flipMatrix() for more info)
            - treeThresh: (relevant) edge weight threshold for adjacency matrix
            - simType:
                - 'cos': cosine similarity
                - 'xor': xor similarity (weights must be discretized)
            focusRank: pivot for flipped Rmsa (only applicable if treeType == 'FRmsa')
        returns:
            - tree: networkx graph object
    '''
    argMax = False
    sim_mat, temp_mat = None, None 

    if simType == 'cos':
        sim_mat = cos_similarity(word_mat)
    elif simType == 'xor':
        sim_mat = xor_similarity(word_mat)

    if treeType is 'msa':
        argMax = True   # maximize similarity edge weight in arboroscence
        temp_mat = sim_mat.copy()
    elif treeType is 'Rmsa':
        rank_mat = rankMatrix(sim_mat)
        temp_mat = rank_mat.copy()
    elif treeType is 'TRmsa':
        rank_mat = rankMatrix(sim_mat)
        temp_mat = rank_mat.T.copy()
    elif treeType is 'FRmsa':
        rank_mat = rankMatrix(sim_mat)
        flipped_rank_mat = flipMatrix(rank_mat,focusRank=focusRank)
        temp_mat = flipped_rank_mat.copy()

    adj_mat = buildAdjMat(temp_mat,thresh=20,mode='abs',reverse=(not argMax))
    g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.DiGraph)
    tree = nx.minimum_spanning_arborescence(g)
    tree = nx.relabel_nodes(tree,indKeys)

    return tree

########################################################################################

def saveTreeDot(tree,fpath='tree.dot'):
    '''
        description:
            - given tree object, saves as dot file
        params: 
            - tree: networkx graph object
            - fpath: output filename for dot file 
        returns:
            None
    '''
    dot = to_pydot(tree)
    write_dot(tree,fpath)    

########################################################################################

def dot2png(dotPath='tree.dot'):
    '''
        description:
            - converts dot file to png (graphviz must be installed)
        params: 
            - dotPath: ['<prefix>.dot'] path to dot file
        returns:
            None; saves png to '<prefix>.png'
    '''
    if not path.exists(dotPath):
        print('path does not exist')
        return None
    if not path.isfile(dotPath):
        print('path not a file')
        return None
    cmd = 'dot -Tpng '+dotPath
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    pngPath = dotPath.rstrip('.dot')+'.png'
    with open(pngPath,'wb') as outfile:
        outfile.write(output)
        outfile.close()
    print('file saved as',pngPath)

########################################################################################

def exportTreeAsCSV(tree,csvOutPath='tree.csv'):
    '''
        description:
            - given tree object, exports edge weights as csv (reqd. for poincare code)
        params: 
            - tree: networkx graph object
            - csvOutPath: output filename for csv file
        returns:
            None
    '''
    csv_columns = ['id1','id2','weight']
    dict_data = list()
    for u in tree:
        for v in tree[u]:
            dict_data.append({'id1':u, 'id2':v, 'weight':1})
    try:
        with open(csvOutPath, 'w') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

########################################################################################
                                # DEMO FUNCTIONS #
########################################################################################

def demo_TRmsa(fpath='/home/kvaditya/GitHubRepos/wiki-news-300d-nouns.txt',pipeline='short'):
    '''
        description:
            - demo function to generate TRmsa tree from given embedding file
        params: 
            - fapth: path to embedding file
            - pipeline:
                - 'short': implement using prebuilt main functions
                - 'long': implement using utility functions
                *NOTE: both methods furnish same results
        returns:
            None
    '''
    if pipeline == 'short':     # using prebuilt main functions

        print('building demo TRmsa using short pipeline')
        vecMat,indKeys,wordKeys = vecFile2vecMat(fpath,norm=True,disc=False)
        tree = vecMat2tree(vecMat,indKeys,treeType='TRmsa',treeThresh=20,simType='cos')
        saveTreeDot(tree,fpath='demo_TRmsa.dot')
        dot2png(dotPath='demo_TRmsa.dot')

        print('process complete')

    elif pipeline == 'long':    # using utility functions

        print('building demo TRmsa using long pipeline')

        if not path.exists(fpath):
            print('error: path does not exist')
            return None
        if not path.isfile(fpath):
            print('error: path not a file')
            return None
        else:

            # step_1) loading embeddings and generating embedding matrix & index-word mappings for the matrix
            emb = load_embedding(fpath,VOCAB=100,typ='w2v')
            indKeys, wordKeys, word_mat = buildWordMat(emb)
            word_mat = normalize(word_mat, axis=0)
            # word_mat = discretize(word_mat, axis=0)  # uncomment to enable discretization

            #step_2) building similarity matrix
            sim_mat = cos_similarity(word_mat)

            #step_3) building rank similarity matrix
            rank_mat = rankMatrix(sim_mat)

            # step_4) building the adjacency matrix, followed by generating the graph
            temp_mat = np.transpose(rank_mat)   # TRmsa
            adj_mat = buildAdjMat(temp_mat,thresh=20,mode='abs',reverse=True)
            g = nx.convert_matrix.from_numpy_array(adj_mat, create_using=nx.DiGraph)

            # step_5) generating the minimum spanning arboroscence of the graph
            tree = nx.minimum_spanning_arborescence(g)
            tree = nx.relabel_nodes(tree,indKeys)  

            # step_6) saving tree to dot file
            dot = to_pydot(tree)
            dotPath = 'demo_TRmsa.dot'
            write_dot(tree,dotPath)

            # step_7) converting dot file to png
            cmd = 'dot -Tpng '+dotPath
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            pngPath = dotPath.rstrip('.dot')+'.png'
            with open(pngPath,'wb') as outfile:
                outfile.write(output)
                outfile.close()
            print('file saved as',pngPath)

            print('process complete')


########################################################################################
                                        #EOF#
########################################################################################



