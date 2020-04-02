text2bin: text2bin.cpp
	g++ -std=c++14 -fsanitize=address -fsanitize=undefined text2bin.cpp -o text2bin

compute-accuracy-topn: compute-accuracy-topn.c
	gcc -lm -pthread  -fuse-ld=gold \
		-march=native  -O3  -funroll-loops  -Wno-unused-result  \
		compute-accuracy-topn.c -o compute-accuracy-topn 
