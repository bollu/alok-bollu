# Download Pretrained Word Embeddings:

FastText: "wiki-news-300d-1M": `wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"`

GloVe 6B: wget -c `"http://nlp.stanford.edu/data/glove.6B.zip"`

GoogleNews-vectors-negative300: wget -c `"https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"`

Or, for all three models, simply run: `bash ./get_embeddings.sh`

We use Gensim in this project, and Gensim only deals with Word2Vec files. Therefore, we need to convert the file format using gensim libraries.

# Building and running `graph.out`:

- We first need a compiled version of `igraph`: follow the following steps, **starting from the current folder**:
```
$ unzip igraph-0.8.2.zip
$ cd igraph-0.8.2
$ ./bootstrap.sh
$ ./configure --prefix=$HOME/.local # installs stuff into your home folder
$ make -j # make -j: build with full parallelism
$ make install # installs the stuff into ~/.local/lib and ~/.local/bin
```

Then run:

```
$ make
$ mkdir -p logs
$ ./graph.out <path-to-fasttext-dump> 10000 &> logs/fasttext-wiki-news-300-scc-vocab=10000.txt
```
# Observations
 - The following instances are resultant components of SCCs formed by the getSCC() method in make-graph.py
 - The initial graph can build in two modes __nsim__(top n similar words set as outward edges) and __thresh__(all words with similarity(normalized dot product) thresh set as outward edges) for each word as a node.
 - To shift b/e the two modes, comment out the alter at lines 22-23 in __make-graph.py__
 - Values of _VOCAB_, _nsim_ or _thresh_ at lines 203,204,and 205 resp. in make-graph.py can be tweaked alter the density of the graph
 
__thresh__ similarity mode
Note: Thresh similarity shows a upper & lower limit of usability. ie. for VOCAB=1000; thresh=0.3 -> single component, thresh=0.7 -> individual components
- Synonym-Antonym pairs and triplets
  - ['correct', 'wrong', 'right'], ['poor', 'bad', 'good'] (thresh=0.65, VOCAB=1000)
  - ['bad', 'good'], ['men', 'women'], ['shows', 'show'], ['private', 'public'] (thresh=0.7, VOCAB=1000)
  - ['current', 'past', 'present'] (thresh=0.60, VOCAB=1000)
  - ['north', 'south'], ['white', 'black'], ['short', 'long'], ['low', 'high'], ['below', 'above'] (thresh=0.65, VOCAB=1000)
  - ['big', 'large', 'small'] (thresh=0.70, VOCAB=1000)
- All word forms of a root {['includes', 'included', 'include', 'including'], ['makes', 'make', 'making', 'made'], ['uses', 'using', 'used', 'use']}
- Compound groups with individully strong subgroups (group-the layman kind ;))
  - ['night', 'hours', 'months', 'week', 'days', 'day', 'month', 'february', 'december', 'november', 'october', 'august', 'july', 'september', 'june', 'january', 'march', 'april', 'may'] (thresh=0.65, VOCAB=1000)
  - ['become', 'became', 'brought', 'saw', 'began', 'did', 'went', 'win', 'lost', 'won', 'taken', 'taking', 'take', 'took', 'came', 'start', 'started', 'getting', 'get', 'got', 'has', 'have', 'been', 'had', 'were', 'was'] (thresh=0.65, VOCAB=1000) // ['taken', 'took', 'taking', 'take'] (thresh=0.70, VOCAB=1000)
- Pronoun grouping
  - ['her', 'she', 'himself', 'him', 'he', 'his'] (thresh=0.65, VOCAB=1000)
  - Pair formation
    - ['he', 'his'], ['she', 'her'], ['we', 'our'] (thresh=0.7, VOCAB=1000)
- General semantic VB & NN grouping
  - ['add', 'adding', 'deleted', 'removed', 'added'] (thresh=0.65, VOCAB=1000)
  - ['friends', 'family', 'families', 'parents', 'child', 'children']
 
__nsim__ similarity mode
 
