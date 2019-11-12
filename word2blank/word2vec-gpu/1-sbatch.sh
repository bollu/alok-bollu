#!/bin/bash
#SBATCH -p long
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v200ng
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
# 1 trains properly
# 2 trains properly
head -c 1000000 text8 > text0
cuda-memcheck ./word2vec -train ../word2vec/text0 -output models/xxxx -cbow 0 -size 20 \
    -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 1 -binary 1 -iter 15 \
    -alpha 0.01

# nvprof ./word2vec -train ../word2vec/text8 -output models/xxxxxx -cbow 0 -size 512 \
#      -window 8 -negative 0 -hs 1 -sample 1e-4 -threads 1 -binary 1 -iter 1 \
#      -alpha 0.01

#  ./word2vec -train ../word2vec/text8 -output models/text8 -cbow 0 -size 512 \
#      -window 8 -negative 0 -hs 1 -sample 1e-4 -threads 1 -binary 1 -iter 30 \
#      -alpha 0.01
# ./1-save-models.sh
# $(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
# ./1-eval.sh models/$NAME.bin
# ./1-save-models.sh
