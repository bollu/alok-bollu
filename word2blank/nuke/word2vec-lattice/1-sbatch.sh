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
make distance
./word2vec -train text0 -cbow 0 -output models/text0 -size 5 -window 8 \
    -negative 25 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 10
# time ./word2vec -train text1 -output models/$GITNAME.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15 
# $(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
# ./1-eval.sh models/$NAME.bin
# ./1-save-models.sh
