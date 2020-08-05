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

using namespace std;


#define ix2(M, i, j) M[i*N+j]


// set p = Q of QR(m)
// void project_q(matrix &m) { };
// void svd(matrix &u, vector &s, matrix &vt, matrix &in) {}

//X, Y: ndim x pdim
float train(arma::Mat<float> X, arma::Mat<float> Y) {
    const int ndim = X.n_rows;
    const int pdim = X.n_cols;

    assert(Y.n_rows == ndim);
    assert(Y.n_cols == pdim);


    arma::Mat<float> dX(ndim, pdim);

    static const int NITER = 1000;
    float L = 0;
    for(int i = 0; i < NITER; ++i) {
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
        const float ALPHA = -1e-3;
        X += dX *ALPHA;

        if ((i % (NITER/10)) == 0) { printf("  - loss: %4d%% | %4.2f\n", 10*(i / (NITER/10)), L); }
    }
    return L;



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

int main() {
    arma::arma_rng::set_seed(0);
    static const int N = 5;
    static const int P = 2;
    for(int i = 0; i < 1000; i++) {
        arma::Mat<float> X(N, P); X.randu();
        arma::Mat<float> Y(N, P); Y.randu();
        float L = train(X, Y);
        printf("final loss of round (%d): %4.2f\n", i+1, L);
    }
    return 0;
}
