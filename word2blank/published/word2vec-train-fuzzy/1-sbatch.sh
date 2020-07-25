#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH -o ./slurm/%j


### SET NAME (NO .bin) ###
NAME=TEXT1
########
########


DATE=`date '+%Y-%m-%d--%H:%M:%S'`
GITNAME=$(git rev-parse --short HEAD)
FOLDERNAME=$GITNAME
mkdir -p models/
mkdir -p slurm/

rm word2vec distance || true
make word2vec
make distance
# ./word2vec -alpha 0.001 -train jabber -cbow 0 -output models/jabber -size 10 -window 4 -negative 1 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15
./word2vec -alpha 0.0005 -train text0 -cbow 0 -output models/text0-size=20  \
    -size 20 -window 30 -negative 0 -hs 0 -sample 1e-4 \
    -threads 1 -binary 1 -iter 10
# time ./word2vec -train text1 -output models/$GITNAME.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 
# $(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
# ./1-eval.sh models/$NAME.bin
# ./1-save-models.sh
