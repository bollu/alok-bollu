#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j

### SET NAME (NO .bin) ###
NAME=text0-regularized-sg-size=10-length=5
########
########


DATE=`date '+%Y-%m-%d--%H:%M:%S'`
GITNAME=$(git rev-parse --short HEAD)
FOLDERNAME=$GITNAME
mkdir -p models/
mkdir -p slurm/

make word2vec
time ./word2vec -train text0 -output models/$GITNAME.bin -cbow 0 -size 10 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15  -debug 2
$(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
./compute-accuracy models/$GITNAME.bin < questions-words.txt > \
    "models/$GITNAME.bin-accuracy.txt"
./compute-accuracy models/$GITNAME.bin < questions-phrases.txt >> \
    "models/$GITNAME.bin-accuracy.txt"
