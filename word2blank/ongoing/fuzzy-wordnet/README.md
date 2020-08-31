# Pipeline
### 1. Preprocessing
  * Calculate raw values from word vectors 
    * Calculate [`sim_mat`](https://github.com/bollu/alok-bollu/blob/3fadea284c3ddf88c8e4be252d8780516ae51a33/word2blank/ongoing/fuzzy-wordnet/main.py#L30) (similarity matrix) using cosine similarity (currently). Plan to change sim measure to KL-Divergence
    * Calculate [`rank_mat[u][v]`](https://github.com/bollu/alok-bollu/blob/3fadea284c3ddf88c8e4be252d8780516ae51a33/word2blank/ongoing/fuzzy-wordnet/main.py#L34) = rank at which u keeps v in terms of similarity
### 2. Building Graph
  * Calculate edge weights from raw values
    * Take `transpose(rank_mat)`: ( tr(rank_mat)[u][v] = rank at which v keeps u ) as Adjacency Mat
    * Remove all weights greater than threshold rank. [`seed=20`](https://github.com/bollu/alok-bollu/blob/cd034f9878835f24fb370a6c1d2fd1a73f2b07df/word2blank/ongoing/fuzzy-wordnet/main.py#L131) used for all current graphs
  * Run Min/Maximum Spanning Arboroscence on graph (Gives directed Tree)
    * Outputs:
      * msa: Maximum Spanning Arboroscence (as edgeWt is similairity, max is considered) [[ link ]](https://imgur.com/a/NlmAZuG)
      * Rmsa: Rank Minimum Spanning Arboroscence [[ link ]](https://imgur.com/a/bK5shrM)
      * TRmsa: Transpose Rank Minimum Spanning Arboroscence [[ link ]](https://imgur.com/a/iYrqTOa)
### 3. Generating Embeddings
  * Load the graph into the [Poincare code](https://github.com/facebookresearch/poincare-embeddings) and generate new embeddings. ([Poincare_paper](https://paperswithcode.com/paper/poincare-embeddings-for-learning-hierarchical?fbclid=IwAR2pGTiV0ais1I9syt_5CP-MGXXwnPSomQSIApSa6syAADHdvu6wbevFRg0))
    * While generating graph, uncomment the [csv generation bit](https://github.com/bollu/alok-bollu/blob/b222a7ccd5fee64c61f103e7d7e4dd5956c12b97/word2blank/ongoing/fuzzy-wordnet/main.py#L141) too in [`main.py`](https://github.com/bollu/alok-bollu/blob/master/word2blank/ongoing/fuzzy-wordnet/main.py) (stores edge list as _.csv_)
    * Set the csv as the input file in [`train-mammals.sh`](https://github.com/facebookresearch/poincare-embeddings/blob/master/train-mammals.sh) and train the embeddings (Spits out torch model)
    * Run [`torch2gensim.py`](https://github.com/bollu/alok-bollu/blob/master/word2blank/ongoing/fuzzy-wordnet/torch2gensim.py) with input file as torch model to get vectors in w2v_format txt file
### 4. Evaluating Embeddings
  * Evaluate poincare embeddings on 
    * Run [`eval.py`](https://github.com/bollu/alok-bollu/blob/master/word2blank/ongoing/fuzzy-wordnet/evaluation/eval.py) with embedding text file set as input. 
    * Output spits out Spearman’s ρ for Lexical Entailment on HYPERLEX ([LEAR_paper](https://arxiv.org/abs/1710.06371))
    * Current ρ on TRmsa graph based embeddings (dim=10): 0.011
