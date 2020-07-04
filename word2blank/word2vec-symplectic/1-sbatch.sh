#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j

set -e
set -o xtrace

mkdir -p models/
mkdir -p slurm/
rm word2vec
gcc -lm -pthread  -march=native -Wall \
    -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold \
    word2vec.c -o word2vec

time ./word2vec -train text8 -output models/size=50.bin \
    -alpha 0.025 -cbow 0 -size 50 -window 8 -negative 25 \
    -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 
