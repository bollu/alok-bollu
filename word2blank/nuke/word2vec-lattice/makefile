CC = gcc
CXX = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O0
RELEASE=-O3
CFLAGS = -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result ${RELEASE} -fuse-ld=gold
CXXFLAGS=${CFLAGS} -std=c++11

all: word2vec word2phrase distance word-analogy compute-accuracy

text0:
	head -c 1000000 text8 > text0

word2vec : word2vec.cpp params.h
	$(CXX) word2vec.cpp -o word2vec $(CXXFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)

distance : distance.cpp params.h
	$(CXX) distance.cpp -o distance $(CXXFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy
