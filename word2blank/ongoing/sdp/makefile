CXX_INCLUDE_PATH=/home/bollu/software/gurobi902/linux64/include/
sdpglove.out: sdpglove.cpp makefile
	g++ sdpglove.cpp -I $(CXX_INCLUDE_PATH) -L$(LD_LIBRARY_PATH) \
		-L/home/bollu/software/gurobi902/linux64/lib/ -lgurobi_c++ -lgurobi90 \
		/home/bollu/software/gurobi902/linux64/lib/libgurobi_c++.a \
		/home/bollu/software/gurobi902/linux64/lib/libgurobi90.so \
		/home/bollu/software/gurobi902/linux64/src/build/libgurobi_c++.a \
		-Wall -Werror \
		-g -fsanitize=address -fsanitize=undefined  -static-libasan \
		-O2  -o sdpglove.out
