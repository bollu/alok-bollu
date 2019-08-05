make
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi

time ./word2vec_two_random_vectors -train text1 -output ./ablation_models/two_random_vectors.bin -cbow 0 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15 ./distance ./ablation_models/two_random_vectors.bin

