#!/bin/bash
set -e
set -o xtrace

./text0-preprocess.sh
./text0-glove.sh
