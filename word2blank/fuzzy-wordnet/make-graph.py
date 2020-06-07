import preprocess

import os
from tqdm import tqdm
import numpy as np

class Graph:
    def __init__(self, emb):
        self.emb = emb
        self.VOCAB = len(emb)
        self.NDIMS = len(list(emb.values())[0])
        return

    def construct_graph(self, nsim):
        graph = np.full(-1, (self.NDIMS, self.NDIMS), dtype=int)
        for w in self.emb:
            w_ix = list(self.emb.keys())index(w)
            sim_list = preprocess.topn_similarity(emb, w, nsim)
            for sim in sim_list:
                (word, wt) = sim
                sim_ix = self.emb.index[word]
                graph[w_ix][sim_ix] = wt
        return graph

class GraphOps:
    def __init__(self, graph):
        sel.graph = graph

    def 

if __name__ == '__main__':

