import preprocess as ps
import os
from tqdm import tqdm
import numpy as np
import sys
import pickle
from collections import OrderedDict

sys.setrecursionlimit(10**7) 

class Graph:
    def __init__(self, emb, nsim):
        self.emb = emb
        self.V = len(emb)
        self.nsim = nsim
        self.NDIMS = len(list(emb.values())[0])
        self.graph = dict()

    def addEdge(self, word):
        self.graph[word] = ps.topn_similarity(self.emb, word, self.nsim)
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
            if(first[v]==-1):
                time, first, lowest, stack = self.findComp(v, time, first, lowest, stack)
                lowest[u] = min(lowest[u], lowest[v])
            elif(v in stack):
                lowest[u] = min(lowest[u], first[v])
        if(first[u]==lowest[u]):
            component = list()
            while(stack[-1]!=u):
                poppedItem = stack.pop()
                component.append(poppedItem)
            poppedItem = stack.pop()
            component.append(poppedItem)
            if(len(component)>0):
                self.compList.append(component)
        return time, first, lowest, stack

    def getSCC(self):
        first = dict.fromkeys(self.graph, -1)
        lowest = dict.fromkeys(self.graph, -1)
        stack = list()
        time = 0
        for u in self.graph:
            if(first[u]==-1):
                time, first, lowest, stack = self.findComp(u, time, first, lowest, stack)

def DFS(graph):
    traversals = list()
    visited = dict.fromkeys(graph, False)
    for root in graph:
        if(not visited[root]):
            visited[root] = True
            ordered_indices, stack = list(), [root]
            while(stack):
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
        if(not visited[root]):
            visited[root] = True
            ordered_indices, stack = list(), [root]
            while(stack):
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
    while(stack):
        u = stack.pop(0)
        for v in graph[u]:
            if(not visited[v]):
                dist[v] = min(dist[u]+1, dist[v]) 
                visited[v] = True
                stack.append(v)
    if(ordered):
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
            if(v==u):
                continue
            u_pli+=(1/allPair[u][v])
        pli[u] = u_pli/(len(graph)-1)

    return pli


if __name__ == '__main__':
    dirname = '../../..'
    # dirname = '../MODELS/'
    fname = 'wiki-news-300d-1M.vec'
    # fname = 'wiki-news-300d-1M.bin'
    NDIMS = 300
    VOCAB = 1000
    nsim = 10
    word_vecs = ps.load_embedding(os.path.join(dirname, fname), VOCAB)
    word_vecs = ps.normalize(word_vecs, 0, NDIMS)
    word_vecs = ps.discretize(word_vecs, 0, NDIMS)
    g = Graph(word_vecs, nsim)
    graph = g.makeGraph()
    g2 = g.removeWts()
    scc = SCC(g2)
    print(scc.compList)

