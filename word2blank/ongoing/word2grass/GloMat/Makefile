CC = gcc
CXX = g++
#For older gcc, use -O3 or -O2 instead of -Ofast
# CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result

# Use -Ofast with caution. It speeds up training, but the checks for NaN will not work
# (-Ofast turns on --fast-math, which turns on -ffinite-math-only,
# which assumes everything is NOT NaN or +-Inf, so checks for NaN always return false
# see https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
# CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wall -Wextra -Wpedantic

CFLAGS = -lm -pthread -O3 -march=native -funroll-loops -Wall -Wextra -Wpedantic
CXXFLAGS = -std=c++14 -lm -O3 -llapack -lblas -lpthread -larmadillo -DARMA_NO_DEBUG -DARMA_MAT_PREALLOC -DARMA_USE_LAPACK -DARMA_USE_BLAS -march=native -Wall -funroll-loops -Wno-unused-result -fuse-ld=gold -I /home/souvik/.local/include -L /home/souvik/.local/lib
BUILDDIR := build
SRCDIR := src
OBJDIR := $(BUILDDIR)
OBJDIRCXX := $(BUILDDIR)

OBJ := $(OBJDIR)/vocab_count.o $(OBJDIR)/cooccur.o $(OBJDIR)/shuffle.o $(OBJDIRCXX)/glove.o
HEADERS := $(SRCDIR)/common.h
HEADERSCXX := $(SRCDIR)/common.h $(SRCDIR)/grad.h
MODULES := $(BUILDDIR)/vocab_count $(BUILDDIR)/cooccur $(BUILDDIR)/shuffle $(BUILDDIR)/glove


all: dir $(OBJ) $(MODULES)
dir :
	mkdir -p $(BUILDDIR)
$(BUILDDIR)/glove : $(OBJDIRCXX)/glove.o $(OBJDIRCXX)/common.o 
	$(CXX) $^ -o $@ $(CXXFLAGS)
$(BUILDDIR)/shuffle : $(OBJDIR)/shuffle.o $(OBJDIR)/common.o
	$(CC) $^ -o $@ $(CFLAGS)
$(BUILDDIR)/cooccur : $(OBJDIR)/cooccur.o $(OBJDIR)/common.o
	$(CC) $^ -o $@ $(CFLAGS)
$(BUILDDIR)/vocab_count : $(OBJDIR)/vocab_count.o $(OBJDIR)/common.o
	$(CC) $^ -o $@ $(CFLAGS)
$(OBJDIR)/%.o : $(SRCDIR)/%.c $(HEADERS)
	$(CC) -c $< -o $@ $(CFLAGS)
$(OBJDIRCXX)/glove.o : $(SRCDIR)/glove.cpp $(HEADERSCXX)
	$(CXX) -c $< -o $@ $(CXXFLAGS)
.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
