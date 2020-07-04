#!/usr/bin/env bash
set -e
set -o xtrace
rm compute-accuracy-topn
rm dump-accuracy-simlex

gcc dump-accuracy-simlex.c -o dump-accuracy-simlex -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold
gcc compute-accuracy-topn.c -o compute-accuracy-topn -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold
MODELPATH=~/work/alok-bollu/models/
SIMLEXPATH=~/work/alok-bollu/word2blank/word2fuzzy/SimLex-999

./compute-accuracy-topn ${MODELPATH}/text8-size=200  < questions-words.txt > w2v-size=200-accuracy.txt
./compute-accuracy-topn ${MODELPATH}/text8-size=100  < questions-words.txt > w2v-size=100-accuracy.txt
./compute-accuracy-topn ${MODELPATH}/text8-size=50  < questions-words.txt >  w2v-size=50-accuracy.txt
./compute-accuracy-topn ${MODELPATH}/text8-size=20  < questions-words.txt >  w2v-size=20-accuracy.txt
./compute-accuracy-topn ${MODELPATH}/text8-size=10  < questions-words.txt >  w2v-size=10-accuracy.txt
./simlex.py  $MODELPATH/text8-size=200 $SIMLEXPATH/SimLex-999.txt > simlex-200.txt
./simlex.py  $MODELPATH/text8-size=100 $SIMLEXPATH/SimLex-999.txt > simlex-100.txt
./simlex.py  $MODELPATH/text8-size=50 $SIMLEXPATH/SimLex-999.txt > simlex-50.txt
./simlex.py  $MODELPATH/text8-size=20 $SIMLEXPATH/SimLex-999.txt > simlex-20.txt
