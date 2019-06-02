#pragma once
#ifndef VEC_H
#define VEC_H
#include <assert.h>
#include <math.h>
#include <stdio.h>
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

__attribute__((constructor)) void initCTable() {
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
        assert(pow2(this->ndims) == len &&
               "dimension number is not powr of 2!");

        int a = posix_memalign((void **)&v, 128, (long long)len * sizeof(real));
        assert(v != nullptr && "memory allocation failed");
        (void)a;
    }

    inline int getlen() const { return len; }
    inline void alloczero(int len) {
        this->len = len;
        this->ndims = log2(len);
        // make sure that the length given is a power of two.
        assert(pow2(this->ndims) == len &&
               "dimension number is not powr of 2!");
        this->v = (real *)calloc(len, sizeof(real));
    }

    // also update partial sums every time set is called.
    // So this is expensive to do.
    inline void set(int i, real val) { v[i] = val; }

    inline real ix(int i) const { return v[i]; }

    inline void fillzero() const {
        for (int i = 0; i < len; ++i) {
            v[i] = 0;
        }
    }

    // return 1?
    inline real lensq() const {
        return dotContainment(*this, /*gradient=*/false, nullptr, nullptr);
    }
    // inline real lensq() const { return 1; }

    inline void normalize() { scale(1.0 / sqrt(lensq()), nullptr); }

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
    // so, for this, we will walk along the tree, and take the dot product
    // of an element of A with the _partial sum of B_ upto that point.
    // eg.
    // A = p + qe_1 + r e_2 + s re_1e_2
    // B = w + xe_1 + ye_2 + ze_1e_2
    // A.b == p (w + x + y + z) + q (x + z) + r (y + z) + s z
    // <scalar> . <anything other than scalar> = 0
    // <full space>  . <anything> = dot product
    inline real dotContainment(const Vec &other, bool grad, float *gbufthis,
                               float *gbufother) const {
        real dot = 0;
        for (unsigned int i = 0; i < pow2(ndims); i++) {
            for (unsigned int j = 0; j < pow2(ndims); j++) {
                // check if J is subset of I
                const bool subset = (j & i) == j;
                if (!subset) continue;
                dot += v[i] * other.v[j];
                if (!grad) continue;

                gbufthis[i] += other.v[j];
                gbufother[j] += this->v[i];
            }
        }

        return dot;
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

// print in little endian: <ABC> = 4A + 2B + C
void printbinary(int v, int ndigits) {
    for (int i = ndigits - 1; i >= 0; i--) {
        printf("%d", (bool)(v & (1 << i)));
    }
}
void printvec(Vec &v, const char *name, real *grad) {
    // number of digits to print == dimension.
    const int ndigits = v.ndims;
    for (int i = 0; i < v.len; ++i) {
        printf("%s", name);
        printf("[");
        printbinary(i, ndigits);
        printf("]");
        printf(": %f", v.v[i]);
        if (grad != nullptr) {
            printf("  ∇");
            printf("%f", grad[i]);
        }
        printf("\n");
    }
}
#endif
