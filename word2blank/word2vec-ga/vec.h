#pragma once
#ifndef VEC_H
#define VEC_H
#include <assert.h>
#include <math.h>
typedef float real;  // Precision of float numbers

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
struct Vec {
    int len;
    int ndims;
    real *v;

   public:
    inline void freemem() { free(v); }

    // return the allocation size needed for a vector of dimension len
    static long int alloc_size_for_dim(int d) { return d * sizeof(real); }
    inline void alloc(int len) {
        this->len = len;
        this->ndims = floor(sqrt((8.0 * len + 1)) / 2.0 - 1.5);
        assert(ndims + ndims * (ndims - 1) / 2 + ndims + 1 <= len);

        int a = posix_memalign((void **)&v, 128, (long long)len * sizeof(real));
        assert(v != nullptr && "memory allocation failed");
        (void)a;
    }

    inline int getlen() const { return len; }
    inline void alloczero(int len) {
        this->len = len;
        this->v = (real *)calloc(len, sizeof(real));
    }

    inline void set(int i, real val) { v[i] = val; }
    inline real ix(int i) const { return v[i]; }
    inline void fillzero() const {
        for (int i = 0; i < len; ++i) v[i] = 0;
    }
    inline real dot(const Vec &v2) const {
        real d = 0;
        for (int i = 0; i < len; ++i) d += v[i] * v2.v[i];
        return d;
    }

    inline real lensq() const { return dot(*this); }

    inline void normalize() { scale(1.0 / sqrt(lensq())); }

    inline void scale(real f) {
        for (int i = 0; i < len; ++i) v[i] *= f;
    }

    inline void accumscaleadd(float f, const Vec &v2) {
        for (int i = 0; i < len; ++i) v[i] += f * v2.v[i];
    }
};

void writevec(FILE *f, Vec &v) {
    for (int a = 0; a < v.getlen(); a++) {
        real r = v.ix(a);
        fwrite(&r, sizeof(real), 1, f);
    }
}

void readvec(FILE *f, Vec &v) {
    for (int a = 0; a < v.getlen(); a++) {
        real r;
        fread(&r, sizeof(real), 1, f);
        v.set(a, r);
    }
}
#endif
