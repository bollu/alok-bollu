#!/usr/bin/env bash
set -e
set -o xtrace

rm jose_50d.bin || true

#gcc dump-accuracy-simlex.c -o dump-accuracy-simlex -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold
gcc compute-accuracy-topn.c -o compute-accuracy-topn -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result -O3 -fuse-ld=gold
g++ -std=c++14 -fsanitize=address -fsanitize=undefined text2bin.cpp -o text2bin

SIMLEXPATH=~/work/alok-bollu/word2blank/word2fuzzy/SimLex-999

./text2bin jose_50d.txt jose-size=50.bin
./compute-accuracy-topn ./jose-size=50.bin 20 < questions-words.txt > jose-size=50-accuracy.txt
# ./compute-accuracy-topn ./jose-size=50.bin 10 < questions-words.txt > jose-size=50-accuracy.txt
