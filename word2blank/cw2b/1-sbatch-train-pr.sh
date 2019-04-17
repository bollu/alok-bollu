#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2m-psr
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j

### SET NAME
NAME="size=10-hfrac=EXPERIMENT"
####

DATE=`date '+%Y-%m-%d--%H:%M:%S'`
GITNAME=$(git rev-parse --short HEAD)
FOLDERNAME=$GITNAME
mkdir -p models/$FOLDERNAME
mkdir -p slurm/

make word2vec
time ./word2vec -train text8 -metrictype pr -frachyperbolic 0.25 -output models/$GITNAME.bin -cbow 0 -size 10 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 30 -binary 1 -iter 15 
./1-save-models.sh
$(cd models; ln -s $GITNAME.bin word2blank/$NAME.bin; cd ../)
./1-eval.sh models/word2blank/$NAME.bin
./1-save-models.sh
