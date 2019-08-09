CC = gcc
DEBUG=-fsanitize=address -fsanitize=undefined -g  
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec word2phrase distance word-analogy compute-accuracy word2vec_two_random_vectors word2vec_one_random_vector word2vec_with_dot_prod_training

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh
word2vec_two_random_vectors : word2vec_two_random_vectors.c
	$(CC) word2vec_two_random_vectors.c -o word2vec_two_random_vectors $(CFLAGS)
word2vec_one_random_vector : word2vec_one_random_vector.c
	$(CC) word2vec_one_random_vector.c -o word2vec_one_random_vector $(CFLAGS)
word2vec_with_dot_prod_training : word2vec_with_dot_prod_training.c
	$(CC) word2vec_with_dot_prod_training.c -o word2vec_with_dot_prod_training $(CFLAGS)

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy
