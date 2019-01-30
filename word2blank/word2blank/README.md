# Requirements
- `python2`
- `pytorch`
- `pip install --user -r requirements.txt`
- `git lfs`

# Submitting a slurm job:
```
# Receive email at the end of the job, job output written to file
# `slurm-<number>.out`
$ sbatch slurm-job.sh
$ tail -f slurm-<number>.out # to view file as it's written to
```

# To use
```
./run.py --help
```

# TODO
- [train vanilla word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
- I might have fucked up by flattening the corpus into sentences. Is this the
  problem? If so, that explains a lot.

# Ideas
- Dot product `<v|` is linear functional (V -> R). Consider using other
  functionals (perhaps non-linear?)
- Dot product `<v|` is a vector field. Consider using other vector fields.
  Interpreting this would be very interesting.
- To train CBOW, don't have `for epoch { for batch in dataset {} } }`. This
  creates weird spikes. Instead, train `for batch in dataset { while loss not stable {train } }`.  
  This should smooth out the "spikiness" of the CBOW training plot.
- Run manifold reduction and see what pops up.
- Read about classification of pseudo reimannian manifolds.

# Code improvements
- use `with autograd.detect_anomaly():` to error on NaN
- Use [gensim keyed vectors](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors)
  for our code
- Use [vecto](https://vecto.readthedocs.io/en/docs/tutorial/index.html) for vector 
space embeddings.
- Use [reimannian SGD](https://arxiv.org/abs/1111.5280)

# Evaluation pipeline
1. Wordnet lookup <-> lookup synset for top-k closest neighbours.
2. 


# To investigate for data handling:
- [sacred](https://sacred.readthedocs.io/en/latest/experiment.html)
- [DVC](https://dvc.org/doc/get-started)

# References
- [ipynb with sample implementation of skip gram](https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.ipynb)
- [CBOW implementation](https://github.com/bastings/nn4nlp2017-code-pytorch/blob/master/01-intro/cbow-pytorch.py)
- [Well written blog post about choices of losses](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#full-softmax)
- [Conformal geometry](https://en.wikipedia.org/wiki/Conformal_geometry)

# Evaluation
- Use [BATS dataset](http://vecto.space/projects/BATS/)
- Use [HyperLex](https://arxiv.org/pdf/1608.02117.pdf)
