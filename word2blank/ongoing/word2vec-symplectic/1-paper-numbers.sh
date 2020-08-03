#!/usr/bin/env bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j

set -e
set -o xtrace

# gcc dump-accuracy-simlex.c -o dump-accuracy-simlex -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold
gcc compute-accuracy.c -o compute-accuracy-topn -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold
MODELPATH=./models
# SIMLEXPATH=~/work/alok-bollu/word2blank/word2fuzzy/SimLex-999

./compute-accuracy ${MODELPATH}/symp-size=400-init0-iters30.bin > accuracies/symp-size=400-init0-iters30.txt
./compute-accuracy ${MODELPATH}/syn1neg-symp-size=400-init0-iters30.bin > accuracies/syn1neg-size=400-init0-iters30.txt
./compute-accuracy ${MODELPATH}/symp-size=400-initrandom-iters30.bin > accuracies/symp-size=400-initrandom-iters30.txt
./compute-accuracy ${MODELPATH}/syn1neg-symp-size=400-initrandom-iters30.bin > accuracies/syn1neg-size=400-initrandom-iters30.txt
# ./simlex.py  $MODELPATH/text8-size=200 $SIMLEXPATH/SimLex-999.txt > simlex-200.txt
# ./simlex.py  $MODELPATH/text8-size=100 $SIMLEXPATH/SimLex-999.txt > simlex-100.txt
# ./simlex.py  $MODELPATH/text8-size=50 $SIMLEXPATH/SimLex-999.txt > simlex-50.txt
# ./simlex.py  $MODELPATH/text8-size=20 $SIMLEXPATH/SimLex-999.txt > simlex-20.txt
