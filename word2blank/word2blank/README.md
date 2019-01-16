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

# To investigate:
- [sacred](https://sacred.readthedocs.io/en/latest/experiment.html)
- [DVC](https://dvc.org/doc/get-started)

# References
- [ipynb with sample implementation](-# https://github.com/jojonki/word2vec-pytorch/blob/master/word2vec.ipynb)
