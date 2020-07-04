#!/usr/bin/env bash
set -e
set -o xtrace

OUTPATH=$(dirname $1)/$(basename $1)-accuracy.txt
# run with <modelpath>
./compute-accuracy $1 < questions-words.txt > $OUTPATH
echo "***PHRASES***" >> $OUTPATH
./compute-accuracy $1 < questions-phrases.txt >> $OUTPATH
