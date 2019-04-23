#pragma once
#ifndef VEC_H
#define VEC_H
#include <assert.h>
#include <math.h>
typedef float real;  // Precision of float numbers

struct Vec {
    int len;
    real *v;

   public:
    inline void freemem() { free(v); }

    // return the allocation size needed for a vector of dimension len
    static long int alloc_size_for_dim(int d) { return d * sizeof(real); }
    inline void alloc(int len) {
        this->len = len;
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
