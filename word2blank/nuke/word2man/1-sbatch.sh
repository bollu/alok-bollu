#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2man
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

rm word2vec 
make word2vec
time ./word2vec -train text8 -output models/text8.bin -cbow 0 -size 100 \
    -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 30 -binary 1 -iter 15 -debug 2
# ./1-save-models.sh
