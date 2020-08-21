CC = gcc
CXX = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O0
RELEASE=-O3
CFLAGS = -llapacke -llapack -lblas -lm -lpthread  -march=native -Wall \
                 -funroll-loops -Wno-unused-result ${RELEASE} -fuse-ld=gold

CFLAGS_arma = -std=c++14 -lm -O3 -llapack -lblas -lpthread -larmadillo -DARMA_NO_DEBUG -DARMA_MAT_PREALLOC -DARMA_USE_LAPACK -DARMA_USE_BLAS -march=native -Wall \
                 -funroll-loops -Wno-unused-result -fuse-ld=gold

all : word2grass compute-accuracy
#distance compute-accuracy

word2grass : word2grass.cpp
		$(CXX) word2grass.cpp -o word2grass $(CFLAGS_arma)

# distance : distance.c
#       $(CC) distance.c -o distance $(CFLAGS)

compute-accuracy : compute-accuracy.cpp
		$(CXX) compute-accuracy.cpp -o compute-accuracy $(CFLAGS_arma) 

clean:
		rm -rf word2grass compute-accuracy
        # distance compute-accuracy

