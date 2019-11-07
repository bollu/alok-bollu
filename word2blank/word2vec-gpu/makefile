CC = nvcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-Xcompiler -fsanitize=address -Xcompiler -fsanitize=undefined -g -O0 -Xcompiler -fuse-ld=gold
RELEASE=-O3 
CFLAGS = -lm -Xcompiler -pthread  -Xcompiler -march=native -Xcompiler -Wall -Xcompiler -funroll-loops -Xcompiler -Wno-unused-result  -O3

all: word2vec word2phrase distance word-analogy compute-accuracy test cli


cli: cli.cpp vec.h linenoise.c
	gcc -c linenoise.c -o linenoise.o $(CFLAGS)
	$(CC) -std=c++11 -c cli.cpp -o cli.o $(CFLAGS)
	$(CC) linenoise.o cli.o -o cli $(CFLAGS)

test: test.cpp vec.h
	$(CC) -std=c++11 test.cpp -o test $(CFLAGS)

word2vec : word2vec.cu vec.h
	$(CC) -std=c++11 word2vec.cu -o word2vec $(CFLAGS)
word2phrase : word2phrase.c vec.h
	$(CC)  word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.cpp vec.h
	$(CC) -std=c++11 distance.cpp -o distance $(CFLAGS)
word-analogy : word-analogy.cpp vec.h
	$(CC) -std=c++11 word-analogy.cpp -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.cpp vec.h
	$(CC) -std=c++11 -fno-exceptions compute-accuracy.cpp -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy
