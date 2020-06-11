import preprocess as ps
import os
from tqdm import tqdm
import numpy as np

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

    def DFS(self, v, visited):
        visited[v] = True
        print(v, end=' ')
        for i in graph[self.graph.index(v)]:
            if not visited[self.graph.index(i)]:
                self.DFS(i, visited)
   
    def fillOrder(self, v, visited, stack):
        visited[v] = True
        for i in graph[self.graph.index(v)]:
            if not visited[self.graph.index(i)]:
                self.fillOrder(i, visited, stack)
        stack.append(v)
        return stack

    def getTranspose(self):
        g = Graph(self.V)
        for i in self.graph:
            for j in graph[i]:
                if j not in g:
                    g.addEdge(j)
        return g

    def getSCC(self):
        stack = list()
        visited = [False]*(self.V)
        for i in range(self.V):
            if not visited[i]:
                stack = self.fillOrder(i, visited, stack)
        g = self.getTranspose(self.graph)
        visited = [False]*(self.V)
        while stack:
            i = stack.pop()
            if visited[i] == False:
                g.DFS(i, visited)
                print('')


if __name__ == '__main__':
    dirname = '../MODELS/'
    fname = 'wiki-news-300d-1M.bin'
    NDIMS = 300
    VOCAB = 500
    nsim = 10
    word_vecs = ps.load_embedding(os.path.join(dirname, fname), VOCAB)
    word_vecs = ps.normalize(word_vecs, 0, NDIMS)
    word_vecs = ps.discretize(word_vecs, 0, NDIMS)
    g = Graph(word_vecs, nsim)
    graph = g.makeGraph()
    g2 = g.removeWts()
    scc = SCC(g2)
    scc.getSCC()
