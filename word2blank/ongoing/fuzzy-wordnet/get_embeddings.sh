#!/bin/bash

echo "Downloading FastText wiki-news-300d-1M"
wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"

echo "Downloading GloVe 6B"
wget -c "http://nlp.stanford.edu/data/glove.6B.zip"

echo "Downloading GoogleNes-vectors-negative300"
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

echo "All models downloaded"
