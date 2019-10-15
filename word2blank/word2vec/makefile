CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O0
RELEASE=-O3
CFLAGS = -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result ${DEBUG} -fuse-ld=gold

all: word2vec word2phrase distance word-analogy compute-accuracy

text0:
	head -c 1000000 text8 > text0

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)

word-analogy-concept : word-analogy-concept.c
	$(CC) word-analogy-concept.c -o word-analogy-concept $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh
dump-vect : dump-vect.c
	$(CC) dump-vect.c -o dump-vect $(CFLAGS)

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy dump-vect
