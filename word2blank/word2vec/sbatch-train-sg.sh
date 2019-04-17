#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2m
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j
DATE=`date '+%Y-%m-%d--%H:%M:%S'`
GITNAME=$(git rev-parse --short HEAD)
FOLDERNAME=$GITNAME---$SLURM_ARRAY_JOB_ID---$DATE
mkdir -p models/$FOLDERNAME
mkdir -p slurm/

make word2vec
time ./word2vec -train text8 -output models/$FOLDERNAME/$GITNAME.bin -cbow 0 -size 50 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 30 -binary 1 -iter 15 
