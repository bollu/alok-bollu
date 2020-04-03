CC = gcc
CXX = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O2
RELEASE=-O3
CFLAGS = -lm -pthread  -march=native -Wall \
		 -funroll-loops -Wno-unused-result ${RELEASE} -fuse-ld=gold

all: word2vec word2phrase distance word-analogy compute-accuracy

bin2text:
	$(CC) bin2text.c -o bin2text

text0:
	head -c 1000000 text8 > text0

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)

distance-context : distance-context.c
	$(CC) distance-context.c -o distance-context $(CFLAGS)

distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)

difference: difference.c
	$(CC) difference.c -o difference $(CFLAGS)

add: add.c
	$(CC) add.c -o add $(CFLAGS)


word-analogy-context : word-analogy-context.c
	$(CC) word-analogy-context.c -o word-analogy-context $(CFLAGS)

word-analogy-concept : word-analogy-concept.c
	$(CC) word-analogy-concept.c -o word-analogy-concept $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

compute-accuracy-topn : compute-accuracy-topn.c
	$(CC) compute-accuracy-topn.c -o compute-accuracy-topn $(CFLAGS)
	chmod +x *.sh

compute-accuracy-ranked : compute-accuracy-ranked.cpp
	$(CC) compute-accuracy-ranked.cpp -o compute-accuracy-ranked $(CFLAGS)
	chmod +x *.sh

word-fuzzy-distance: word-fuzzy-distance.c
	$(CC) word-fuzzy-distance.c -o word-fuzzy-distance $(CFLAGS)

dump-vect : dump-vect.c
	$(CC) dump-vect.c -o dump-vect $(CFLAGS)

dump-accuracy-simlex : dump-accuracy-simlex.c
	$(CC) dump-accuracy-simlex.c -o dump-accuracy-simlex $(CFLAGS)


compute-composition: compute-composition.cpp
	$(CC) compute-composition.cpp -o compute-composition $(CFLAGS)

cli: cli.cpp  linenoise.c
	$(CC) -c linenoise.c -o linenoise.o $(CFLAGS)
	$(CXX) -std=c++11 -c cli.cpp -o cli.o $(CFLAGS)
	$(CXX)  linenoise.o cli.o -o cli $(CFLAGS)


clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy dump-vect
