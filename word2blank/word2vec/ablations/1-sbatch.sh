#!/bin/bash
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name=w2v
#SBATCH --mail-type=END
#SBATCH -o ./slurm/%j

make
if [ ! -e text8  ]; then
      wget http://mattmahoney.net/dc/text8.zip -O text8.gz
        gzip -d text8.gz -f
fi

### SET NAME (NO .bin) ###
NAME=ablations
########
########


mkdir -p ablation_models/
mkdir -p slurm/

make word2vec word2vec_two_random_vectors                                        
time ./word2vec -train text8 -output ./ablation_models/naive_text8.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 40 -binary 1 -    iter 15
time ./word2vec_two_random_vectors -train text8 -output ./ablation_models/two_random_vectors_text8.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sam    ple 1e-4 -threads 40 -binary 1 -iter 15
./compute-accuracy ./ablation_models/naive_text8.bin < ../questions-words.txt > ./ablation_models/naive_text8_accuracies.txt
./compute-accuracy ./ablation_models/two_random_vectors_text8.bin < ../questions-words.txt > ./ablation_models/two_random_vectors_text8_accuracies.txt        
