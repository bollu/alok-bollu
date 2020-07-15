.PHONY: graph.out # rebuild graph.out each time, regardless of whether it exists or not.
# debug flags
# CXXFLAGS=-O0 =fsanitize=address -fsanitize=undefined
# release flags
CXXFLAGS=-O3
CXX=g++

graph.out: graph.cpp
	$(CXX) graph.cpp -g -o graph.out -std=c++14 \
		-Wno-unused-result \
		-I${HOME}/.local/include/igraph/  \
		-L${HOME}/.local/lib/ -ligraph \
		-funroll-loops -march=native -fuse-ld=gold  -fopenmp -ffast-math \
		${CXXFLAGS}
