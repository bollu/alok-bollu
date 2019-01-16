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

# To investigate for data handling:
- [sacred](https://sacred.readthedocs.io/en/latest/experiment.html)
- [DVC](https://dvc.org/doc/get-started)

# References
- [ipynb with sample implementation of skip gram](https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.ipynb)
- [CBOW implementation](https://github.com/bastings/nn4nlp2017-code-pytorch/blob/master/01-intro/cbow-pytorch.py)
