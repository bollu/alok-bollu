CC = gcc
CXX = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O0
RELEASE=-O3
CFLAGS = -llapcke -llapack -lblas -lm -lpthread  -march=native -Wall \
		 -funroll-loops -Wno-unused-result ${RELEASE} -fuse-ld=gold

all : word2grass distance

word2grass : word2grass.c
	$(CC) word2grass.c -o word2grass $(CFLAGS)

distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)

clean:
	rm -rf word2grass distance 
