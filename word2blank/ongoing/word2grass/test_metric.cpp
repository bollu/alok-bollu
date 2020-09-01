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
const long long int N = 3;

void train(arma::mat current, arma::mat target)
{
    const long long int ndim = current.n_rows;
    const long long int pdim = current.n_cols;

    assert((long long int)target.n_rows == ndim);
    assert((long long int)target.n_cols == pdim);

    static const int NITER = 10000;
    const double ALPHA = 2e-2;
    double loss = 0.0;
    arma::mat dcurrent(N,P); 
    arma::mat dtarget(N,P); 
    for ( long long int i = 0; i < NITER; i++ )
    {   
        dcurrent.zeros(); dtarget.zeros();
        //getDotAndGradients_chordalfrobenius(current, target, loss, dcurrent, dtarget);
        //getDotAndGradients_binetcauchy(current, target, loss, dcurrent, dtarget);
        //getDotAndGradients_martin(current, target, loss, dcurrent, dtarget);
        getDotAndGradients_fubinistudy(current, target, loss, dcurrent, dtarget);
        cout << "Loss at iteration " << i << " is " << loss << "\n";
        current -= dcurrent*ALPHA;
        current = arma::orth(current);
        cout << "CURRENT SUBSPACE:\n" << current;
    }

}

int main()
{
    arma::mat current(N,P); current.randu();
    current = arma::orth(current);
    arma::mat target(N,P); target.randu();
    target = arma::orth(target);
    cout << "TARGET SUBSPACE:\n" << target ;
    train(current, target);
    return 0;
}