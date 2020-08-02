#!/usr/bin/env bash
set -e
set -o xtrace
gcc -lm -pthread  -march=native -Wall \
    -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold \
    word2vec.c -o word2vec

time ./word2vec -train ~/alok-bollu/word2blank/utilities/text8 \
     -output symp-size4-text8.bin     \
     -alpha 0.001 -cbow 0 -size 4 -window 8 -negative 25    \
      -hs 0 -sample 1e-4 -threads 24 -binary 1 -iter 15
