CC = gcc
CXX = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
DEBUG=-fsanitize=address -fsanitize=undefined -g -O0
RELEASE=-O3
mFLAGS = -llapacke -llapack -lblas -lm -lpthread  -march=native -Wall \
                 -funroll-loops -Wno-unused-result ${RELEASE} -fuse-ld=gold

CFLAGS_arma = -std=c++14 -lm -O3 -L ${HOME}/OpenBLAS-0.3.13 -lopenblas -lpthread -DARMA_NO_DEBUG -DARMA_MAT_PREALLOC -DARMA_USE_LAPACK -DARMA_USE_BLAS -DARMA_DONT_USE_WRAPPER -march=native -Wall \
                 -funroll-loops -Wno-unused-result -fuse-ld=gold \
                 -I ${HOME}/armadillo-9.900.2/include -L ${HOME}/.local/lib
all : word2grass compute-accuracy distance simlex-accuracy word2grass_ada
#distance compute-accuracy

word2grass : word2grass.cpp
		$(CXX) word2grass.cpp -o word2grass $(CFLAGS_arma)

jose : jose.cpp
		$(CXX) jose.cpp -o jose $(CFLAGS_arma)

word2grass_jose : word2grass_jose.cpp
		$(CXX) word2grass_jose.cpp -o word2grass_jose $(CFLAGS_arma)

word2steif_jose : word2steif_jose.cpp
		$(CXX) word2steif_jose.cpp -o word2steif_jose $(CFLAGS_arma)

word2steif : word2steif.cpp
		$(CXX) word2steif.cpp -o word2steif $(CFLAGS_arma)

compute-accuracy : compute-accuracy.cpp
		$(CXX) compute-accuracy.cpp -o compute-accuracy $(CFLAGS_arma) 

simlex-accuracy : simlex-accuracy.cpp
		$(CXX) simlex-accuracy.cpp -o simlex-accuracy $(CFLAGS_arma)

distance : distance.cpp
		$(CXX) distance.cpp -o distance $(CFLAGS_arma)
bollu-analogy : bollu-analogy.cpp
		# $(CXX) analogy.cpp -o analogy $(CFLAGS_arma)
		$(CXX) bollu-analogy.cpp -o bollu-analogy \
			 -I ${HOME}/.local/include -L ${HOME}/.local/lib \
			 -llapack -lblas -lpthread -larmadillo -lm \
			 -DARMA_USE_LAPACK -DARMA_USE_BLAS  \
			 -std=c++14 -lm -fuse-ld=gold -fsanitize=address -fsanitize=undefined

test_metric : test_metric.cpp
		$(CXX) test_metric.cpp -o test_metric $(CFLAGS_arma)
clean:
		-rm -rf word2grass compute-accuracy simlex-accuracy metric test_metric distance word2grass_ada

backprop-bollu: backprop-bollu.cpp makefile 
		$(CXX) backprop-bollu.cpp -o backprop-bollu $(CFLAGS_arma)
