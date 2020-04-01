#!/usr/bin/env bash
rm compute-accuracy-topn
rm dump-accuracy-simlex
make dump-accuracy-simlex
make compute-accuracy-topn 
MODELPATH=~/work/alok-bollu/models/
SIMLEXPATH=~/work/alok-bollu/word2blank/word2fuzzy/SimLex-999

# ./compute-accuracy-topn ${MODELPATH}/text8-size=50  < questions-words.txt > fuzzy-size=50-accuracy-top5.txt
# ./compute-accuracy-topn ${MODELPATH}/text8-size=20  < questions-words.txt > fuzzy-size=20-accuracy-top5.txt
# ./compute-accuracy-topn ${MODELPATH}/text8-size=10  < questions-words.txt > fuzzy-size=10-accuracy-top5.txt
./simlex.py  $MODELPATH/text8-size=200 $SIMLEXPATH/SimLex-999.txt > simlex-200.txt
./simlex.py  $MODELPATH/text8-size=100 $SIMLEXPATH/SimLex-999.txt > simlex-100.txt
./simlex.py  $MODELPATH/text8-size=50 $SIMLEXPATH/SimLex-999.txt > simlex-50.txt
./simlex.py  $MODELPATH/text8-size=20 $SIMLEXPATH/SimLex-999.txt > simlex-20.txt
