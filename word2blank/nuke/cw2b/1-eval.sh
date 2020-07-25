#!/usr/bin/env bash
set -e
set -o xtrace
# run with <modelpath>
# 0 = usemetric. That is, perform vector dot products with regular dot product
./compute-accuracy $1 0 < questions-words.txt > $(dirname $1)/$(basename $1)-accuracy.txt
echo "***PHRASE***" >> $(dirname $1)/$(basename $1)-accuracy.txt
./compute-accuracy $1 0 < questions-phrases.txt >> $(dirname $1)/$(basename $1)-accuracy.txt
