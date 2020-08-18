CC = gcc
CXX = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O2
RELEASE=-O3
CFLAGS = -lm -pthread  -march=native -Wall \
		 -funroll-loops -Wno-unused-result ${RELEASE} -fuse-ld=gold

GPU_DEBUG=-Xcompiler -fsanitize=address -Xcompiler -fsanitize=undefined -g -O0 -Xcompiler -fuse-ld=gold
GPU_RELEASE=-O3

all: word2vec word2phrase distance word-analogy compute-accuracy

bin2text:
	$(CC) bin2text.c -o bin2text

text0:
	head -c 1000000 text8 > text0

gpu-word2vec: word2vec.cu makefile
	nvcc word2vec.cu -o gpu-word2vec -lm \
        -Xcompiler -pthread  \
        -Xcompiler -march=native \
        -Xcompiler -Wall -Xcompiler \
        -funroll-loops -std=c++11 \
		-Xcompiler -Wall -Xcompiler -Werror \
        -Xcompiler -Wno-unused-result  $(GPU_DEBUG)

word2vec : word2vec.c makefile
	$(CC) word2vec.c -o word2vec $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)

distance-context : distance-context.c
	$(CC) distance-context.c -o distance-context $(CFLAGS)

bizarre : bizarre.c
	$(CC) bizarre.c -o bizarre $(CFLAGS)

word2vec-principal-compute-accuracy : word2vec-principal-compute-accuracy.cpp
	$(CC) word2vec-principal-compute-accuracy.cpp -o word2vec-principal-compute-accuracy $(CFLAGS)
word2vec-principal-analogy : word2vec-principal-analogy.cpp
	$(CC) word2vec-principal-analogy.cpp -o word2vec-principal-analogy $(CFLAGS)
principal-distance : principal-distance.cpp
	$(CC) principal-distance.cpp -o principal-distance $(CFLAGS)
word2vec-principal-compute-accuracy-topn : word2vec-principal-angle-compute-accuracy-topn.cpp
	$(CC) word2vec-principal-angle-compute-accuracy-topn.cpp -o word2vec-compute-accuracy-topn $(CFLAGS)
	chmod +x *.sh
word2vec-principal-compute-accuracy-ranked : word2vec-principal-angle-compute-accuracy-ranked.cpp
	$(CC) word2vec-principal-angle-compute-accuracy-ranked.cpp -o word2vec-compute-accuracy-ranked $(CFLAGS)
	chmod +x *.sh

hamiltonian: hamiltonian.c
	$(CC) hamiltonian.c -o hamiltonian $(CFLAGS)
	chmod +x *.sh

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


word-fuzzy-distance: word-fuzzy-distance.c
	$(CC) word-fuzzy-distance.c -o word-fuzzy-distance $(CFLAGS)

dump-vect : dump-vect.c
	$(CC) dump-vect.c -o dump-vect $(CFLAGS)

dump-accuracy-simlex : dump-accuracy-simlex.c
	$(CC) dump-accuracy-simlex.c -o dump-accuracy-simlex $(CFLAGS)


compute-composition: compute-composition.cpp
	$(CC) -O2 compute-composition.cpp -o compute-composition -lm  -Wno-unused-result

word2vec-compute-accuracy : word2vec-compute-accuracy.cpp
	$(CC) word2vec-compute-accuracy.cpp -o word2vec-compute-accuracy $(CFLAGS)
	chmod +x *.sh

cli: makefile cli.cpp  linenoise.c
	$(CC) -c linenoise.c -o linenoise.o $(CFLAGS)
	$(CXX) -std=c++11 -c cli.cpp -o cli.o $(CFLAGS)
	$(CXX)  linenoise.o cli.o -o cli $(CFLAGS)


clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy* dump-vect word2vec-principal-analogy principal-distance word2vec-compute-accuracy*
