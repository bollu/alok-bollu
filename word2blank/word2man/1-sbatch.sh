#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j

### SET NAME (NO .bin) ###
NAME=TEXT1
########
########


DATE=`date '+%Y-%m-%d--%H:%M:%S'`
GITNAME=$(git rev-parse --short HEAD)
FOLDERNAME=$GITNAME
mkdir -p models/
mkdir -p slurm/

make word2vec
time ./word2vec -train text1 -output models/text1.bin -cbow 0 -size 10 -window 4 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 -debug 2
# ./1-save-models.sh
