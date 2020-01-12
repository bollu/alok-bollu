CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O0
RELEASE=-O3
CFLAGS = -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result ${RELEASE} -fuse-ld=gold

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

word-intersection : word-intersection.c
	$(CC) word-intersection.c -o word-intersection $(CFLAGS)

word-complement : word-complement.c
	$(CC) word-complement.c -o word-complement $(CFLAGS)

word-union : word-union.c
	$(CC) word-union.c -o word-union $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

compute-accuracy-topn : compute-accuracy-topn.c
	$(CC) compute-accuracy-topn.c -o compute-accuracy-topn $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy
