CC = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O0
RELEASE=-O3
CFLAGS = -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result ${RELEASE} -L$(HOME)/.local/lib -I$(HOME)/.local/include/ -ladept

all: word2vec word2phrase distance word-analogy compute-accuracy

word2vec : word2vec.cpp vec.h
	$(CC) -std=c++11 word2vec.cpp -o word2vec $(CFLAGS) 
word2phrase : word2phrase.c vec.h
	$(CC)  word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.cpp vec.h
	$(CC) -std=c++11  distance.cpp -o distance $(CFLAGS)
word-analogy : word-analogy.cpp vec.h
	$(CC) -std=c++11 word-analogy.cpp -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.cpp vec.h
	$(CC) -std=c++11 compute-accuracy.cpp -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy
