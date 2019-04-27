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

// dumb implementation of log2
int log2(int n) {
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

static const int MAXC = 1000;
// table containing binomial coefficients C[n][r]
int C[MAXC][MAXC];

// init the table of C[n][r]
void initCTable() {
    C[0][0] = 1;
    C[1][0] = C[1][1] = 1;

    for (int n = 1; n < MAXC; n++) C[n][0] = 1;
    for (int n = 2; n < MAXC; ++n) {
        for (int r = 1; r <= n; ++r) {
            C[n][r] = C[n][r - 1] + C[n - 1][r - 1];
        }
    }
}

// dumb encoding of GA. uses log2(n)elements.
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
        this->ndims = log2(len);
        // make sure that the length given is a power of two.
        assert(pow2(ndims) == len && "dimension number is not powr of 2!");

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

    // return 1?
    // inline real lensq() const { return dot(*this); }
    inline real lensq() const { return 1; }

    inline void normalize() { scale(1.0 / sqrt(lensq())); }

    inline void scale(real f) {
        for (int i = 0; i < len; ++i) v[i] *= f;
    }

    inline void accumscaleadd(float f, const Vec &v2) {
        for (int i = 0; i < len; ++i) v[i] += f * v2.v[i];
    }

    // scalar product is useless!
    // https://arxiv.org/pdf/1205.5935.pdf (Geometric Algebra: Eric Chisolm)
    // Start with (Eqn 156)
    // A * B = sum_r Ar * Br where Ar, Br are multivectors
    // Notice Ar * Br =  < Ar^dagger leftdot Br>
    // For multivectors:
    //     Ar^dagger = (e0e1e2...er)^\dagger =
    //          (-1)^{r(r-1)/2} (e0e1e2...er) (Eqn 128)
    //
    //  Plugging back in:
    //  Ar * Br = (-1)^{r(r-1)/2} (Ar leftdot Br)
    //
    //  Now note that Ar, Br have the same grade, so they have the same
    //  number of dimensions.
    //
    // We now proceed to show how to evaluate Ar^dagger leftdot Br
    // - Ar^dagger leftdot Br = (-1)^{r(r-1)/2} (Ar leftdot Br)
    //   if Ar has a vector that is orthogonal to Br, then Ar leftdot Br = 0.
    //   Notice that they are both of the same grade. So, if they do not
    //   have the *same* basis vectors, then Ar *will* contain a vector
    //   orthogonal to Br. This means that Ar and Br must share the same basis
    //   vectors.
    //
    //- Now, let us consider Ar and Br have the same basis. So,
    //  Ar = a e0 e1 ... en
    //  Ar^dagger = a en e_n-1 ... e0
    //  Br = b e0 e1 .. en
    //
    //  Ar^dagger Br = (a en en-1 ... e1 e0) (b e0 e1 ... en)
    //               = a b(en en-1 ... e1 e0) (e0 e1 ... en)
    //               = a b e0^2(en en-1 ... e1  e1 ... en) = ...
    //               = a b
    //
    // So, the scalar product is _literally_ a fucking dot product? what a
    // joke. I need a different type of product.
    //
    //
    // Fuck this, I'm going to perform the equivalent of "roll my own crypto".
    // If a  multivector A  is a subset of a multivector B, then we compute
    // coeff(a) * coeff(b). This will probably make the multiplication
    // __non symmetric__. For example,
    // 1. scalar * space = scalar (since the space contains the scalar)
    // 2. space * scalar = 0 (since the scalar does NOT contain the space)
    inline real dotContainment(const Vec &v2) {
        real dot = 0;
        // the r in nCr
        for (int sd = 0; sd < this->ndims; ++sd) {
            // number of elements in this s dimension = 2^s
            const int ns = sd == 0 ? 1 : C[ndims][sd];
            // 1 + nC0 + nC1 + .. nCs.
            const int sbase = sd == 0 ? 0 : pow2(sd - 1);
            for (int rd = 0; rd <= sd; ++rd) {
                const int nr = rd == 0 ? 1 : C[ndims][rd];
                const int rbase = rd == 0 ? 0 : pow2(rd - 1);

                for (int s = 0; s < ns; ++s) {
                    for (int r = 0; r < nr; ++r) {
                        // r \subset s
                        // r / (r \cap s) == emptyset
                        // r ^ (r & s) == emptyset
                        if ((r ^ (r & s)) != 0) continue;
                        dot += v[sbase + s] * v[rbase + r];
                    }
                }
            }
        }
        return dot;
    }

    // what should this do?
    // this is returning the scalar product!
    // inline real dot(const Vec &v2) const {
    //     real d = 0;
    //     for (int i = 0; i < len; ++i) d += v[i] * v2.v[i];
    //     return d;
    // }
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
