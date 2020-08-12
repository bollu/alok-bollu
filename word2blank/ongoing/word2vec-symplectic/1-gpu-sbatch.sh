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
rm gpu-word2vec || true
nvcc word2vec.cu -o gpu-word2vec -lm \
    -Xcompiler -pthread  \
    -Xcompiler -march=native \
    -Xcompiler -Wall -Xcompiler \
    -funroll-loops \
    -Xcompiler -Wno-unused-result  -O3 \
    -std=c++11 -Xcompiler -Wall -Xcompiler -Werror

time ./gpu-word2vec -train ../../utilities/text8 -output models/symp-size=100-iters=20 \
    -alpha 0.025 -cbow 0 -size 100 -window 8 -negative 25 \
    -hs 0 -sample 1e-4 -threads 1 -binary 1 -iter 20
