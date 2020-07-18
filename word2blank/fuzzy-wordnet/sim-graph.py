import preprocess as ps
from os import path
from tqdm import tqdm
import numpy as np
import sys
import pickle as pkl
from collections import OrderedDict
#from numba import jit, cuda

sys.setrecursionlimit(10**7) 

class Graph:
    def __init__(self, emb, nsim, thresh):
        self.emb = emb
        self.V = len(emb)
        self.nsim = nsim
        self.thresh = thresh
        self.NDIMS = len(list(emb.values())[0])
        self.graph = dict()

    def addEdge(self, word):
        self.graph[word] = ps.thresh_similarity(self.emb, word, self.thresh)
        #self.graph[word] = ps.topn_similarity(self.emb, word, self.nsim)
        return self.graph[word]

    def makeGraph(self):
        for w1 in tqdm(list(self.emb.keys())):
            self.graph[w1] = self.addEdge(w1)
        return self.graph

    def removeWts(self):
        wtlessGraph = dict()
        for node, sim in self.graph.items():
            wtless = [s[0] for s in sim]
            wtlessGraph[node] = wtless
        return wtlessGraph

class SCC:
    def __init__(self, graph):
        self.graph = graph
        self.V = len(graph)
        self.compList = list()

    def findComp(self, u, time, first, lowest, stack):
        time += 1
        first[u] = lowest[u] = time
        stack.append(u)
        for v in self.graph[u]:
            if (first[v] == -1):
                time, first, lowest, stack = self.findComp(v, time, first, lowest, stack)
                lowest[u] = min(lowest[u], lowest[v])
            elif (v in stack):
                lowest[u] = min(lowest[u], first[v])
        if (first[u] == lowest[u]):
            component = list()
            while (stack[-1] != u):
                poppedItem = stack.pop()
                component.append(poppedItem)
            poppedItem = stack.pop()
            component.append(poppedItem)
            if (len(component) > 0):
                self.compList.append(component)
        return time, first, lowest, stack

    def getSCC(self):
        first = dict.fromkeys(self.graph, -1)
        lowest = dict.fromkeys(self.graph, -1)
        stack = list()
        time = 0
        for u in self.graph:
            if (first[u] == -1):
                time, first, lowest, stack = self.findComp(u, time, first, lowest, stack)

def DFS(graph):
    traversals = list()
    visited = dict.fromkeys(graph, False)
    for root in graph:
        if (not visited[root]):
            visited[root] = True
            ordered_indices, stack = list(), [root]
            while (stack):
                u = stack.pop()
                ordered_indices.append(u)
                for v in graph[u]:
                    if(not visited[v]):
                        stack.append(v)
                        visited[v] = True
            traversals.append(ordered_indices)
    return traversals

def BFS(graph):
    traversals = list()
    visited = dict.fromkeys(graph, False)
    for root in graph:
        if (not visited[root]):
            visited[root] = True
            ordered_indices, stack = list(), [root]
            while (stack):
                u = stack.pop(0)
                ordered_indices.append(u)
                for v in graph[u]:
                    if(not visited[v]):
                        stack.append(v)
                        visited[v] = True
            traversals.append(ordered_indices)
    return traversals

def shortestPath(graph, root, ordered=False):
    dist, visited = dict.fromkeys(graph, float('inf')), dict.fromkeys(graph, False)
    dist[root] = 0
    stack = [root]
    while (stack):
        u = stack.pop(0)
        for v in graph[u]:
            if (not visited[v]):
                dist[v] = min(dist[u]+1, dist[v]) 
                visited[v] = True
                stack.append(v)
    if (ordered):
        return OrderedDict(sorted(dist.items(), key=lambda x: x[1], reverse=False))
    else:
        return dist

def allPairShortestPath(graph):   

    pathdict = dict()
    for u in graph:
        pathdict[u] = shortestPath(graph, u)
    return pathdict

def pathLenIndex(graph, indistance=False):
    ''' 
        indistance: lenghth of shortest inward path to u from v
        outdistance: length of shortest outward path from u to v
    '''
    pli = dict()
    allPair = allPairShortestPath(graph)

    if(indistance):
        revAllPair = dict()
        for u in allPair:
            if(u not in allPair):
                revAllPair[u] = dict()
            for v in allPair[u]:
                if(v not in revAllPair):
                    revAllPair[v] = dict()
                revAllPair[v][u] = allPair[u][v]
        allPair = revAllPair

    u_pli = 0
    for u in graph:
        for v in graph[u]:
            if (v==u):
                continue
            u_pli+=(1/allPair[u][v])
        pli[u] = u_pli/(len(graph)-1)

    return pli

def calc_dist(g2):
    dist = dict.fromkeys(g2, dict.fromkeys(g2, -1))
    for w1 in g2:
        for w2, d in g2[w1]:
            if(w1 is not w2) and (dist[w1][w2]<0):
                dist[w1][w2] = d
                dist[w2][w1] = d
    return dist

def inequality_satisfied(dlist):
    result = True
    for i in range(3):
        d1, d2, d3 = dlist[i], dlist[(i+1)%3], dlist[(i+2)%3]
        if(d1>d2+d3 or d1<abs(d2-d3)):
            result = False
            break 
    return result


def check_traingles(dist):
    record = []
    visited = dict.fromkeys(dist, dict.fromkeys(dist, dict.fromkeys(dist, False)))
    for w1 in tqdm(dist):
        for w2 in dist:
            if(w1==w2):
                continue
            for w3 in dist:
                if(w1==w3 or w2==w3):
                    continue
                if(visited[w1][w2][w3]):
                    continue
                if( not inequality_satisfied([dist[w1][w2],dist[w2][w3],dist[w3][w1]])):
                    print(w1,w2,w3)
                    record.append([w1,dist[w1][w2],w2,dist[w2][w3],w3,dist[w3][w1],w1])
    return record


if __name__ == '__main__':
    dirname = ''#'../../..'
    # dirname = '../MODELS/'
    fname = 'wiki-news-300d-1M.vec'
    # fname = 'wiki-news-300d-1M.bin'
    NDIMS = 300
    VOCAB = 1000
    nsim = 10
    thresh = 0.65
    word_vecs = ps.load_embedding(path.join(dirname, fname), VOCAB)
    word_vecs = ps.normalize(word_vecs, 0, NDIMS)
    word_vecs = ps.discretize(word_vecs, 0, NDIMS)
    g = Graph(word_vecs, nsim, thresh)
    g2 = g.makeGraph()
    print("graph made")
    # g2 = g.removeWts()
    mode = 2
    #null mode
    if(mode==0):
        mode=input("Enter mode: 1.Prt Grph  2.Prt SCC  3.Chck Tri Ineq") 

    #Print Graph
    if(mode==1):
        print(g2)

    #Print SCC
    if(mode==2):
        scc = SCC(g2)
        scc.getSCC()
        scc.compList.sort(key=len)
        for comp in scc.compList:
            print(comp)

    #Confirm triangle inequality
    if(mode==3):
        dist = calc_dist(g2)
        record = check_traingles(dist)	#stores non-conformant triplets  
        print("dist calced")
        print(record)
        print("end")
       
