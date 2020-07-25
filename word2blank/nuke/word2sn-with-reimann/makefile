CC = gcc
NVCC=nvcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O0
RELEASE=-O3
CFLAGS = -lm -pthread  -march=native -Wall -funroll-loops -Wno-unused-result ${RELEASE} -fuse-ld=gold
NVCCFLAGS = -lm -Xcompiler -pthread  -Xcompiler -march=native -Xcompiler \
						-Wall -Xcompiler -funroll-loops -Xcompiler -Wno-unused-result -O3

all: word2vec word2phrase distance word-analogy compute-accuracy

text0:
	head -c 1000000 text8 > text0

word2vec : word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)

word2vecgpu : word2vec.cu
	$(NVCC) word2vec.cu -o word2vecgpu $(NVCCFLAGS)

word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

clean:
	rm -rf word2vec word2phrase distance word-analogy compute-accuracy
