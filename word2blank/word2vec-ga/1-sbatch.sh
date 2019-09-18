#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j

### SET NAME (NO .bin) ###
NAME=size8-window8-negative25-iter20
########
########


DATE=`date '+%Y-%m-%d--%H:%M:%S'`
GITNAME=$(git rev-parse --short HEAD)
FOLDERNAME=$GITNAME
mkdir -p models/
mkdir -p slurm/

make word2vec
head -c 1000000 text8 > text0
time ./word2vec -train text8 -output models/text8-size=8-unweighed.bin -cbow 0 -size 8  \
    -window 8 -negative 15 -hs 0 -sample 1e-4 -threads 38 -binary 1 -iter 50 \
    -alpha 0.01
# ./1-save-models.sh
# $(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
# ./1-eval.sh models/$NAME.bin
# ./1-save-models.sh
