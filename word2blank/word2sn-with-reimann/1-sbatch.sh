#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
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

make word2vec
./word2vec -train text8 -cbow 0 -output models/text8-size=200-window=8-negative=25  -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 
# ./word2vec -train text0 -cbow 0 -output models/0  -size 10 -window 4 -negative 10 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 
# time ./word2vec -train text1 -output models/$GITNAME.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 
# $(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
# ./1-eval.sh models/$NAME.bin
# ./1-save-models.sh
