// http://arma.sourceforge.net/docs.html
// https://j-towns.github.io/papers/svd-derivative.pdf
// Estimating-th-Jacobian-of-the-Singular-Value-Decomposition
//     https://hal.inria.fr/inria-00072686/document
// clean slate implementation of backprop.
#include <iostream>
#include <stdio.h>
#include <assert.h>
// ARMA_NO_DEBUG
#include <armadillo>
#include <vector>

using namespace std;


#define ix2(M, i, j) M[i*N+j]

// set p = Q of QR(m)
// void project_q(matrix &m) { };
// void svd(matrix &u, vector &s, matrix &vt, matrix &in) {}

//X, Y: ndim x pdim
pair<arma::Mat<float>, float>
train(arma::Mat<float> X, arma::Mat<float> Y) {
    const int ndim = X.n_rows;
    const int pdim = X.n_cols;

    assert(Y.n_rows == ndim);
    assert(Y.n_cols == pdim);


    arma::Mat<float> dX(ndim, pdim);

    static const int NITER = 10000;
    float L = 0;
    for(int i = 0; i < NITER; ++i) {
        arma::Mat<float> XQ, XR;
        arma::Mat<float> P = X.t() * Y;

        arma::Col<float> S;
        arma::Mat<float> U, V;

        svd(U,S,V,P);

        L  = arma::accu(S % S);
        dX.zeros(); // clear the gradient
        for(int f = 0; f < ndim; ++f) { 
            for(int g = 0; g < pdim; ++g) { 
                for(int gamma = 0; gamma < pdim; ++gamma) { 
                    for(int eps = 0; eps < pdim; ++eps) { 
                        dX(f, g) += 2 * S(gamma) * U(g, gamma) * V(eps, gamma) * Y(f, eps);
                    }
                }
            }
        }
        const float ALPHA = 1e-3;
        X += dX *ALPHA;
        X = arma::orth(X);

        if ((i % (NITER/10)) == 0) { printf("  - loss: %4d%% | %4.2f\n", 10*(i / (NITER/10)), L); }
    }
    return make_pair(X, L);



    /*
    // X, Y: ndimxpdim
    matrix P = new_matrix(p, p); // P for product X^T Y; P = pdimxpdim
    ortho U = new_matrix(p, p), VT = new_matrix(p, p);
    vector S  = new_vector[m]; // diagonal, vector.
    float L = 1e9;

    project_q(Y);


    //P1| P[b][c] = Σα (X^T)[b][α] Y[α][c]
    //    P[b][c] = Σα X[α][b] Y[α][c] | α:ndim
    //
    //P2| U, S, VT = SVD(P) 
    //
    //
    //P3| loss = Σβ S[β] S[β]   | β:pdim
    //
    // dloss/dX[fg] = Σγδε dloss/dS[γ] * dS[γ]/dP[δε] * dP[δε]/dX[fg]
    //
    //G1| dP[δ][ε]/dX[fg] 
    //     = d(Σα X[α][δ]Y[α][ε])/dX[fg]
    //     = Σα Y[α][ε] Dirac(α, f) Dirac(δ, g)
    //         { contract }
    //     = Y[f][ε] Dirac(δ, g)
    //
    //
    //G2| dS[γ]/dP[δ][ε] 
    //     = u[δ][γ] v[ε][γ] { eqn 7 from estimating the jacobian of the SVD }
    //
    //G3| dloss/dS[γ] 
    //     = d(Σβ S[β] S[β])/dS[γ] 
    //     = Σβ 2S[β] Dirac(γ, β)
    //       { contract }
    //     = 2S[γ]   
    //
    //
    // total:
    //   dloss/dX[fg] 
    //       = Σγδε dloss/dS[γ] * dS[γ]/dP[δε]  * dP[δε]/dX[fg]
    //       = Σγδε 2S[γ] u[δ][γ] v[ε][γ] Y[f][ε] Dirac(δ, g) 
    //         { contract }
    //       = Σγε 2S[γ] u[g][γ] v[ε][γ] Y[f][ε]
    //       = Σγε 2S[γ] u[g][γ] v^T[γ][ε] Y[f][ε]
    
    
    while (L > 1e-2) {
        // project X using QR.
        project_q(X);
        for(int b = 0; b < pdim; ++b) {
            for(int c = 0; c < pdim; ++c) {
                ix2(P, b, c) = 0;
                for(int alpha = 0; alpha < pdim; ++alpha) {
                    ix2(P, b, c) += ix2(X, alpha, b) * ix2(Y, alpha, c);
                }
            }
        }

        svd(U, S, VT, P);

        L = 0;
        for(int f = 0; f < ndim; ++f) { 
            for(int g = 0; g < pdim; ++g) { 
                float dX = 0;
                for(int gamma = 0; gamma < pdim; ++gamma) { 
                    for(int eps = 0; eps < pdim; ++eps) { 
                        dX += 2 * S[gamma] * ix2(U, g, gamma) * ix2(VT, gamma, eps) * ix2(Y, f, eps);
                    }
                }
                const float ALPHA = 1e-3;
                ix2(X, f, g) += -1 * ALPHA * dX;
            }
        }

        printf("loss: %4.2d\n", L);
    }
    */
    
}


std::vector<std::vector<bool>> combinations(int N, int P) {
    std::vector<std::vector<bool>> vs;
    std::vector<bool> v(N);
    std::fill(v.begin(), v.begin() + P, true);

    do {
        vs.push_back(v);
    } while (std::prev_permutation(v.begin(), v.end()));
    return vs;
}

/*
// ======= Generating combinations ===============
// let us keep things sane, can't get too large numbers.
assert(N < 10);
int comb[N+1][N+1];

for(int n = 0; n <= N; ++n) {
    comb[n][0] = 1;
    for (int r = 1; r <= n; ++r) {
        comb[n][r] = comb[n-1][r-1] + comb[n-1][r];
    }
    for(int r = n+1; r < N; ++r) { comb[n][r] = 0; }
}

for(int n = 0; n <= N; ++n) {
    for(int r = 0; r <= N; ++r) {
        printf("%dC%d %d   ", n, r, comb[n][r]);
    }
    printf("\n");
}

// Iterating over all subspaces of nCp
// https://stackoverflow.com/questions/9430568/generating-combinations-in-c
*/

int main() {
    arma::arma_rng::set_seed(0);
    srand(0);
    static const int N = 5;
    static const int P = 2;

    const vector<vector<bool>> combs = combinations(N, P);
    int i = 0;
    printf("total size: |%d|\n", int(combs.size()));
    for(vector<bool> comb : combs) {
        cout << "====\n";
        cout << "|";
        for(int c = 0; c < comb.size(); ++c) { cout << (comb[c] ? "x" : "-"); }
        cout << "|\n";
        arma::Mat<float> Y(N, P); Y.zeros();
        for(int c = 0, ix = 0; c < N; ++c) { if (comb[c]) { Y(c, ix++) = 1; } }


        arma::Mat<float> X0(N, P); X0.randu(); X0 = arma::orth(X0);
        arma::Mat<float> Xn(N, P);
        cout << "Y:\n" << Y << "\nX0: \n" << X0;
        float L = 42;
        tie(Xn, L) = train(X0, Y);
        Xn = arma::orth(Xn);
        static const float CLEAN_THRESHOLD=1e-4;

        // TODO: replace with call to clean();
        for(int n = 0; n < N; ++n) { 
            for(int p = 0; p < P; ++p) { 
                if (fabs(Xn(n, p)) < CLEAN_THRESHOLD) { Xn(n, p) = 0; }
            }
        }
        cout << "Xn: \n" << Xn;
        printf("final loss of round (%d/%d): %4.2f\n", 1+i++, int(combs.size()), L);
    }
    return 0;
}
