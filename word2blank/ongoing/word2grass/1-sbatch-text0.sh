#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2gr
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j

set -e
set -o xtrace

mkdir -p models/
mkdir -p models/standard
mkdir -p models/iters
mkdir -p models/windowsize
mkdir -p slurm/

rm word2grass || true
make word2grass

time ./word2grass -train ../../utilities/text0 -output models/grass-text0.bin \
    -alpha 0.1 -cbow 0 -n 19 -p 5 -window 4 \
    -hs 1 -sample 1e-4 -threads 30 -binary 1 -iter 3
