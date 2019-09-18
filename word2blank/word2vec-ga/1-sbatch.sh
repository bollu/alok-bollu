#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j


mkdir -p models/
mkdir -p slurm/

make word2vec
head -c 1000000 text8 > text0
time ./word2vec -train text8 -output models/text8-size=8-window=4-neg=30-iter=30-1f691d.bin -cbow 0 -size 8  \
    -window 4 -negative 30 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 30 \
    -alpha 0.01
# ./1-save-models.sh
# $(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
# ./1-eval.sh models/$NAME.bin
# ./1-save-models.sh
