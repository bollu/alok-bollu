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

time ./word2grass -train ../../utilities/text0  -output 75grass.bin -alpha 0.1 -cbow 0 -n 75 -p 1 -negative 3 -window 5 -hs 0 -sample 1e-3 -threads 80 -binary 1 -iter 30
