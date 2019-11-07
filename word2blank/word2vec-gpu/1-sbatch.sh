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

rm word2vec
make word2vec
head -c 1000000 text8 > text0
# cuda-memcheck ./word2vec -train text0 -output models/xxxx -cbow 0 -size 8 \
#     -window 8 -negative 0 -hs 1 -sample 1e-4 -threads 1 -binary 1 -iter 15 \
#     -alpha 0.01
# 
 ./word2vec -train ../word2vec/text0 -output models/xxxx -cbow 0 -size 64 \
     -window 8 -negative 0 -hs 1 -sample 1e-4 -threads 1 -binary 1 -iter 30 \
     -alpha 0.01
# ./1-save-models.sh
# $(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
# ./1-eval.sh models/$NAME.bin
# ./1-save-models.sh
