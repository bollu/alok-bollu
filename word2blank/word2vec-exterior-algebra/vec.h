#pragma once
#ifndef VEC_H
#define VEC_H
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cblas.h>

typedef float real;  // Precision of float numbers

template <typename T>
T min(T x, T y) {
   return x < y ? x : y;
}

// let the dimensionality of the space be n. We have:
// 1 -- 0D component
// n -- 1D components
// n (n - 1)/2 -- 2D components
// n -- (n-1)D components
// 1 -- nD  component
// In [28]: (n, y) = symbols("n y"); p = 1+n+n*(n-1)/2+n+1-y; solve(p, n)
// Out[28]: [-sqrt(8*y + 1)/2 - 3/2, sqrt(8*y + 1)/2 - 3/2]

// if dim > 3, then the 2D and (n-1)D objects are different objects
// so roughly, we can support floor(sqrt(8y-7)/2 - 3.0/2) number of dimensions
// For length:
// len = 100, we get 12 dimensions as the answer.
// 1 + 12 + 12*11/2 + 12 + 1 = 24 + 66 + 1 = 92
//
// We can probably look into fancier structurings of the subspaces with respect
// to sparsity by perhaps pulling tricks from compressed sensing? This is a
// longshot, though.

// dumb implementation of log2
int log2(int n) {
  if (n == 0) return 0;

   int l = 0;
   int i = 1;
   while (i < n) {
      i = i * 2;
      l += 1;
   }
   return l;
}
int pow2(int n) {
   int p = 1;
   while (n-- > 0) p *= 2;
   return p;
}

static const int MAXC = 20;
// table containing binomial coefficients C[n][r]
int C[MAXC][MAXC];

// init the table of C[n][r]

// NOTE: check recurrene
__attribute__((constructor)) void initCTable() {
   C[0][0] = 1;
   C[1][0] = C[1][1] = 1;

   for (int n = 1; n < MAXC; n++) C[n][0] = 1;
   for (int n = 1; n < MAXC; n++) C[n][n] = 1;
   for (int n = 2; n < MAXC; ++n) {
      for (int r = 1; r < n; ++r) {
         C[n][r] = C[n][r - 1] + C[n - 1][r - 1];
      }
   }
}

int dirac(int i, int j) {
    return i == j ? 1 : 0;
}

int det(int i1, int i2, int j1, int j2) {
    return i1 * j2 - i2 * j1;
}

// r = n * n * sizeof(real)
// n = log2 d
void setupDotContainmentMat(int len, real *r) {
    int sqn = sqrt(len);
    assert (sqn * sqn == len);

    for(int i1 = 0; i1 < sqn; ++i1) {
        for(int j1 = 0; j1 < sqn; ++j1) {
            for(int i2 = 0; i2 < sqn; ++i2) {
                for(int j2 = 0; j2 < sqn; ++j2) {
                    const int v = det(dirac(i1, i2), dirac(i1, j2), dirac(j1, i2), dirac(j1, j2));
                    r[(i1 + sqn * j1) + len * (i2 + sqn *j2)] = v;
                }
            }
        }
    }


    printf("\n===\n");
    printf("sqn: %d\n", sqn);
    for(int i = 0; i < len; ++i) {
        for(int j = 0; j < len; ++j) {
            printf("%d ", int(r[i * len + j]));
        }
        printf("\n");
    }
    printf("===\n");

}

// compute X^T A Y with BLAS
// X: Nx1
// A: NxN
// Y: Nx1
// gx: Nx1
// gy: Nx1
// Note that xTA is OPTIONAL, but Ay is _definitely not_.
float mulQuadForm(int dim, const float *x, const float *A, const float *y, float *Ay, float *xTA) {
    float xAy = 0;
    // A = DIM x DIM
    // y = DIM x 1
    // Ay = DIM x DIM x DIM x 1 = DIM x 1
     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,                      
            dim, 1, dim,                                                        
            1, // alpha                                                         
            A, dim,                                                             
            y, 1,                                                             
            0,  // beta                                                                 
            Ay, 1);                                                           
                                                                                
    // xT Ay
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            1, 1, dim,
            1, // alpha
            x, 1,
            Ay, 1,
            0,  // beta
            &xAy, 1); 


    if (xTA) {
                                                                                
        // X = DIM x 1
        // xT = 1 x DIM
        // A = DIM x DIM
        // xTA = 1 x DIM x DIM x DIM = 1 x DIM
        cblas_sgemv(CblasRowMajor, CblasTrans,
                dim, dim, 
                1, //alpha
                A, dim,
                x, 1,
                0, // beta
                xTA, 1);
    }



    return xAy;

}


// dumb encoding of GA. uses log2(n)elements.
struct Vec {
   int len;
   real *v;

  public:
   inline void freemem() { free(v); }
   Vec() : len(0), v(nullptr){}
   Vec(const Vec &other) : len(other.len), v(other.v) {};


   // return the allocation size needed for a vector of dimension len
   static long int alloc_size_for_dim(int d) { return d * sizeof(real); }
   inline void alloc(int len) {
      this->len = len;
      // make sure that the length given is a power of two.
      int a = posix_memalign((void **)&v, 128, (long long)len * sizeof(real));
      assert(v != nullptr && "memory allocation failed");
      (void)a;
   }


   inline int getlen() const { return len; }
   inline void alloczero(int len) {
      this->len = len;
      // make sure that the length given is a power of two.
      this->v = (real *)calloc(len, sizeof(real));
   }


   inline void fillzero() const {
      for (int i = 0; i < len; ++i) {
         v[i] = 0;
      }
   }


   inline void scale(real f, real *gbuf) {
      for (int i = 0; i < len; ++i) v[i] *= f;
      // z = x * const
      // dz/dt = const * dx/dt
      if (gbuf == nullptr) return;
      for (int i = 0; i < len; ++i) gbuf[i] *= f;
   }

   inline void accumscaleadd(real f, const Vec &other) {
      for (int i = 0; i < len; ++i) v[i] += f * other.v[i];
   }
};

void writevec(FILE *f, Vec &v) {
   for (int a = 0; a < v.getlen(); a++) {
      real r = v.v[a];
      fwrite(&r, sizeof(real), 1, f);
   }
}

void readvec(FILE *f, Vec &v) {
   for (int a = 0; a < v.getlen(); a++) {
      real r;
      fread(&r, sizeof(real), 1, f);
      v.v[a] =r;
   }
}

// print in little endian: <ABC> = 4A + 2B + C
void printbinary(int v, int ndigits) {
   for (int i = ndigits - 1; i >= 0; i--) {
      printf("%d", (bool)(v & (1 << i)));
   }
}

#endif // VEC_H
