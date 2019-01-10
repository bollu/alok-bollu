# Requirements
- `python2`
- `pytorch`
- `pip install --user -r requirements.txt`

# Submitting a slurm job:
```
# Receive email at the end of the job, job output written to file
# `slurm-<number>.out`
$ sbatch slurm-job.sh
$ tail -f slurm-<number>.out # to view file as it's written to
```
