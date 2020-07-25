#!/bin/bash
#SBATCH -p long
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j
#SBATCH -A nlp


rm word2vec distance
mkdir -p models/
mkdir -p slurm/

make word2vec distance
head -c 1000000 text8 > text0
time ./word2vec -train text0 -output models/text0 -cbow 0 -size 8 \
    -window 25 -negative 0 -hs 1 -sample 1e-4 -threads 40 -binary 1 -iter 30 \
    -alpha 0.02
# ./1-save-models.sh
# $(cd models; ln -s $GITNAME.bin $NAME.bin; cd ../)
# ./1-eval.sh models/$NAME.bin
# ./1-save-models.sh
