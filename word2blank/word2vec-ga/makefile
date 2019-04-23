CC = clang
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec word2phrase distance word-analogy compute-accuracy

word2vec : word2vec.cpp
	$(CC) -std=c++11 word2vec.cpp -o word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC)  word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.cpp
	$(CC) -std=c++11 -fno-exceptions distance.cpp -o distance $(CFLAGS)
word-analogy : word-analogy.cpp
	$(CC) -std=c++11 word-analogy.cpp -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.cpp
	$(CC) -std=c++11 -fno-exceptions compute-accuracy.cpp -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy
