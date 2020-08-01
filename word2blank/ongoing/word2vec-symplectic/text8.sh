#!/usr/bin/env bash
set -e
set -o xtrace
gcc -lm -pthread  -march=native -Wall \
    -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold \
    word2vec.c -o word2vec

time ./word2vec -train ~/alok-bollu/word2blank/utilities/text8 \
     -output models/symp-text8-size=10-iter=20.bin     \
     -alpha 0.025 -cbow 0 -size 200 -window 8 -negative 25    \
      -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 20
