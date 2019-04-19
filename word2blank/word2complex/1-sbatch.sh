#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2cplx
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j

### SET NAME (NO .bin) ###
NAME=complex-size=100
########
########


DATE=`date '+%Y-%m-%d--%H:%M:%S'`
GITNAME=$(git rev-parse --short HEAD)
FOLDERNAME=$GITNAME
mkdir -p models/
mkdir -p slurm/

set -e
set -o xtrace
make word2vec
time ./word2vec -train text8 -output models/$GITNAME.bin -cbow 0 -size 100 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15 
./1-save-models.sh
$(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
./1-eval.sh models/$NAME.bin
./1-save-models.sh
