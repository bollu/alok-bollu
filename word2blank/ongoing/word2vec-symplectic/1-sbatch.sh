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
mkdir -p models/standard
mkdir -p models/iters
mkdir -p models/windowsize
mkdir -p slurm/
rm word2vec || true
gcc -lm -pthread  -march=native -Wall \
    -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold \
    word2vec.c -o word2vec

time ./word2vec -train ../../utilities/text8 -output models/standard/symp-size=200.bin \
    -alpha 0.025 -cbow 0 -size 200 -window 8 -negative 25 \
    -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 

time ./word2vec -train ../../utilities/text8 -output models/iters/symp-size=200iters40.bin \
    -alpha 0.025 -cbow 0 -size 200 -window 8 -negative 25 \
    -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 40 

time ./word2vec -train ../../utilities/text8 -output models/windowsize/symp-size=200window6.bin \
    -alpha 0.025 -cbow 0 -size 200 -window 6 -negative 25 \
    -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 

time ./word2vec -train ../../utilities/text8 -output models/windowsize/symp-size=200negative10.bin \
    -alpha 0.025 -cbow 0 -size 200 -window 8 -negative 10 \
    -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 
