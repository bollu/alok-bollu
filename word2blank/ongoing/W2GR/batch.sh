#!/bin/bash
#SBATCH -n 16
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mincpus=40
#SBATCH --gres=gpu:4
#SBATCH --mail-user=souvik.banerjee@research.iiit.ac.in
#SBATCH --mail-type=ALL

time ./jose -train ~/Spherical-Text-Embedding/datasets/aclImdb/text.txt -doc-output models/imdb-emb1_4.txt -size 100 -p 4 -l 1 -alpha 0.04 -margin 0.15 -window 5 -negative 2 -sample 1e-3 -iter 60 -threads 40 -binary 0 -min_count 5
time ./jose -train ~/Spherical-Text-Embedding/datasets/aclImdb/text.txt -doc-output models/imdb-emb1_2.txt -size 100 -p 2 -l 1 -alpha 0.04 -margin 0.15 -window 5 -negative 2 -sample 1e-3 -iter 60 -threads 40 -binary 0 -min_count 5
time ./jose -train ~/Spherical-Text-Embedding/datasets/aclImdb/text.txt -doc-output models/imdb-emb1_1.txt -size 100 -p 1 -l 1 -alpha 0.04 -margin 0.15 -window 5 -negative 2 -sample 1e-3 -iter 60 -threads 40 -binary 0 -min_count 5
time ./jose -train ~/Spherical-Text-Embedding/datasets/aclImdb/text.txt -doc-output models/imdb75-emb1_3.txt -size 75 -p 3 -l 1 -alpha 0.04 -margin 0.15 -window 5 -negative 2 -sample 1e-3 -iter 35 -threads 40 -binary 0 -min_count 5
