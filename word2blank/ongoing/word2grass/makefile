CC = gcc
CXX = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O0
RELEASE=-O3
mFLAGS = -llapacke -llapack -lblas -lm -lpthread  -march=native -Wall \
                 -funroll-loops -Wno-unused-result ${RELEASE} -fuse-ld=gold

CFLAGS_arma = -std=c++14 -lm -O3 -llapack -lblas -lpthread -larmadillo -DARMA_NO_DEBUG -DARMA_MAT_PREALLOC -DARMA_USE_LAPACK -DARMA_USE_BLAS -march=native -Wall \
                 -funroll-loops -Wno-unused-result -fuse-ld=gold \
                 -I ${HOME}/.local/include -L ${HOME}/.local/lib
all : word2grass compute-accuracy simlex-accuracy metric
#distance compute-accuracy

word2grass : word2grass.cpp
		$(CXX) word2grass.cpp -o word2grass $(CFLAGS_arma)


compute-accuracy : compute-accuracy.cpp
		$(CXX) compute-accuracy.cpp -o compute-accuracy $(CFLAGS_arma) 

simlex-accuracy : simlex-accuracy.cpp
		$(CXX) simlex-accuracy.cpp -o simlex-accuracy $(CFLAGS_arma)

distance : distance.cpp
		$(CXX) distance.cpp -o distance $(CFLAGS_arma)
test_metric : test_metric.cpp
		$(CXX) test_metric.cpp -o test_metric $(CFLAGS_arma)
clean:
		-rm -rf word2grass compute-accuracy simlex-accuracy metric test_metric distance

# backprop-bollu.out: backprop-bollu.cpp makefile
# 	g++ -std=c++14 -fsanitize=address -fsanitize=undefined \
# 		backprop-bollu.cpp -o backprop-bollu.out \
# 		-larmadillo

