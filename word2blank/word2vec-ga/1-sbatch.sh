#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j


mkdir -p models/
mkdir -p slurm/

make word2vec distance
head -c 1000000 text8 > text0
time ./word2vec -train text8 -output models/text8 -cbow 0 -size 64 \
    -window 4 -negative 0 -hs 1 -sample 1e-4 -threads 40 -binary 1 -iter 15 \
    -alpha 0.03
# ./1-save-models.sh
# $(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
# ./1-eval.sh models/$NAME.bin
# ./1-save-models.sh
