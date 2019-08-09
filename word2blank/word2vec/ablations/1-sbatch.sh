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

make word2vec_one_random_vector  word2vec_with_dot_prod_training

time ./word2vec_one_random_vector -train text8 -output ./ablation_models/one_random_vector_text8.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15
./compute-accuracy ./ablation_models/one_random_vector_text8.bin < ../questions-words.txt > ./ablation_models/one_random_vector_text8_accuracies.txt

time ./word2vec_with_dot_prod_training -train text8 -output ./ablation_models/with_dot_prod_training_text8.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 40 -binary 1 -iter 15
./compute-accuracy ./ablation_models/with_dot_prod_training_text8.bin < ../questions-words.txt > ./ablation_models/with_dot_prod_training_text8_accuracies.txt
