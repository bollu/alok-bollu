#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <armadillo>
#include <vector>
#include <string>
#include "vec.h"

using namespace std;

const long long int P = 2;
const long long int N = 4;

//train for projection metric
void train_proj(arma::mat current, arma::mat target)
{
    const long long int ndim = current.n_rows;
    const long long int pdim = current.n_cols;

    assert((long long int)target.n_rows == ndim);
    assert((long long int)target.n_cols == pdim);

    static const int NITER = 10000;
    const double ALPHA = 1e-3;
    double loss = 0.0;
    arma::mat dcurrent(N,P); dcurrent.zeros();
    arma::mat dtarget(N,P); dtarget.zeros();
    for ( long long int i = 0; i < NITER; i++ )
    {   
        getDotAndGradients_projection(current, target, loss, dcurrent, dtarget);
        cout << "Loss at iteration " << i << " is " << loss << "\n";
        current -= dcurrent*ALPHA;
        current = arma::orth(current);
        cout << "CURRENT SUBSPACE:\n" << current;
    }

}

//train for geodesic distance
// void train_geo(arma::mat current, arma::mat target)
// {
//     const long long int ndim = current.n_rows;
//     const long long int pdim = current.n_cols;

//     assert((long long int)target.n_rows == ndim);
//     assert((long long int)target.n_cols == pdim);

//     static const int NITER = 1000;
//     const double ALPHA = 1e-3;
//     double loss = 0.0;
//     arma::mat dcurrent(N,P); dcurrent.zeros();
//     arma::mat dtarget(N,P); dtarget.zeros();
//     for ( long long int i = 0; i < NITER; i++ )
//     {   
//         getDotAndGradients_geodesic(current, target, loss, dcurrent, dtarget);
//         cout << "LOSS AT iteration " << i << "=" << loss << "\n";
//         current -= dcurrent*ALPHA;
//         current = arma::orth(current);
//         cout << "CURRENT SUBSPACE:\n" << current;
//     }

// }

int main()
{
    arma::mat current(N,P); current.randu();
    arma::mat B(N,P); B.randu();
    arma::mat target = arma::orth(B);
    cout << "TARGET SUBSPACE:\n" << target ;
    
    train_proj(current, target);
    return 0;
}