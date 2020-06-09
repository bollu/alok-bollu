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

class GraphOps:
    def __init__(self, graph):
        self.graph = graph

if __name__ == '__main__':
    dirname = '../MODELS/'
    fname = 'wiki-news-300d-1M.bin'
    NDIMS = 300
    VOCAB = 50000
    nsim = 50
    word_vecs = ps.load_embedding(os.path.join(dirname, fname), VOCAB)
    word_vecs = ps.normalize(word_vecs, 0, NDIMS)
    word_vecs = ps.discretize(word_vecs, 0, NDIMS)
    g = Graph(word_vecs, nsim)
    graph = g.makeGraph()
    for key, val in graph.items():
        print(key + '\t' + str(val))
