#!/bin/bash
set -e
set -o xtrace

make clean
./text0-preprocess.sh
./text0-glove.sh
