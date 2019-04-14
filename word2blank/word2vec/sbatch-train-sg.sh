#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2m
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j
make word2vec
time ./word2vec -train text8 -output vectors-sg.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
