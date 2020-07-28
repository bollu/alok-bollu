#!/usr/bin/env bash

time ./word2vec -train ~/alok-bollu/word2blank/utilities/text0 \
     -output models/symp-text0-size=100-iter=40.bin     \
     -alpha 0.025 -cbow 0 -size 100 -window 8 -negative 25    \
      -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 40 
