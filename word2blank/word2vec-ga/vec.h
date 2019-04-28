#pragma once
#ifndef VEC_H
#define VEC_H
#include <adept/Stack.h>
#include <adept/scalar_shortcuts.h>
#include <assert.h>
#include <math.h>
using adept::adouble;
typedef float real;  // Precision of float numbers
#define min(x, y) ((x) < (y) ? (x) : (y))

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
long long log2(long long n) {
    long long l = 0;
    long long i = 1;
    while (i < n) {
        i = i * 2;
        l += 1;
    }
    return l;
}
long long pow2(long long n) {
    long long p = 1;
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
template <typename T>
struct VecT {
    int len;
    int ndims;
    T *v;

   public:
    inline void freemem() { free(v); }

    // return the allocation size needed for a vector of dimension len
    static long int alloc_size_for_dim(int d) { return d * sizeof(real); }
    inline void alloc(int len) {
        // set len to be larger so we can start indexing from 1.
        this->ndims = log2((long long)len);
        // make sure that the length given is a power of two.
        assert(pow2(ndims) == len && "dimension number is not powr of 2!");

        this->len = len + 2;
        // allocate len+2 so we can 1-index
        v = new T[len + 2];
        // int a = posix_memalign((void **)&v, 128,
        //                        ((long long)len + 2) * sizeof(real));
        // assert(v != nullptr && "memory allocation failed");
        // (void)a;
    }

    inline int getlen() const { return len; }
    inline void alloczero(int len) {
        alloc(len);
        for (int i = 0; i < this->len; ++i) v[i] = 0;
    }

    inline void set(int i, T val) { v[i] = val; }
    inline T ix(int i) const { return v[i]; }
    inline void fillzero() const {
        for (int i = 0; i < len; ++i) v[i] = 0;
    }

    // return 1?
    // inline real lensq() const { return dot(*this); }
    inline T lensq() const { return T(1); }

    inline void normalize() { scale(1.0 / sqrt(lensq())); }

    inline void scale(T f) {
        for (int i = 0; i <= len; ++i) v[i] *= f;
    }

    inline void accumscaleadd(T f, const VecT<T> &v2) {
        for (int i = 0; i <= len; ++i) v[i] += f * v2.v[i];
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
    inline T dotContainment(const VecT<T> &v2) {
        T dot = T(0);
        // the r in nCr
        // printf("===\n");
        for (int sd = 0; sd < ndims; ++sd) {
            // printf("---\n");
            // printf("sd: %4d\n", sd);
            // number of elements in this s dimension = 2^s
            const int ns = C[ndims][sd];
            // 1 + nC0 + nC1 + .. nCs.
            const int sbase = pow2(sd);
            for (int rd = 0; rd <= sd; ++rd) {
                // printf("\trd: %4d\n", rd);
                const int nr = C[ndims][rd];
                const int rbase = pow2(rd);

                for (int s = 0; s < ns; ++s) {
                    // printf("\t\ts: %4d |six: %4d\n", s, sbase + s);
                    for (int r = 0; r < nr; ++r) {
                        // printf("\t\t\tr: %4d |rix: %4d\n", r, rbase + r);
                        // r \subset s
                        // r / (r \cap s) == emptyset
                        // r ^ (r & s) == emptyset

                        // HACK: we need a condition that checks
                        // if they share bases!
                        // if ((r ^ (r & s)) != 0) continue;
                        dot += this->v[rbase + r] * v2.v[sbase + s];
                    }
                }
            }
        }
        // getchar();
        return dot;
    }

    // what should this do?
    // this is returning the scalar product!
    inline T dot(const VecT<T> &v2) const {
        T d = T(0);
        for (int i = 0; i < len; ++i) d += v[i] * v2.v[i];
        return d;
    }
};

using Vec = VecT<real>;
using VecDiff = VecT<adept::adouble>;

void printvec(const Vec &v, int n) {
    printf("|");
    for (int a = 0; a < min(n, v.getlen()); a++) {
        printf("%5.2f", v.ix(a));
        if (a != min(n, v.getlen()) - 1) printf(" ");
    }
    printf("|\n");
}

void printvecdiff(const VecDiff &v, int n) {
    printf("|");
    for (int a = 0; a < min(n, v.getlen()); a++) {
        printf("%5.2f", (real)v.ix(a).value());
        if (a != min(n, v.getlen()) - 1) printf(" ");
    }
    printf("|\n");
}

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
void writevecdiff(FILE *f, VecDiff &v) {
    for (int a = 0; a < v.getlen(); a++) {
        real r = v.ix(a).value();
        fwrite(&r, sizeof(real), 1, f);
    }
}

void readvecdiff(FILE *f, VecDiff &v) {
    for (int a = 0; a < v.getlen(); a++) {
        real r;
        fread(&r, sizeof(real), 1, f);
        v.set(a, r);
    }
}
#endif
