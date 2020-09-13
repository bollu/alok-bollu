#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <armadillo>
#include <vector>
#include <string>
#include "grad.h"

using namespace std;
using namespace std::chrono;

const long long int P = 3;
const long long int N = 4;
const long long int label = sqrt(P);
void train(arma::mat current, arma::mat target)
{
    const long long int ndim = current.n_rows;
    const long long int pdim = current.n_cols;

    assert((long long int)target.n_rows == ndim);
    assert((long long int)target.n_cols == pdim);
    long long int i = 0;
    const double ALPHA = 1e-3;
    double distance = 0.0, loss = 0.0;
    arma::mat dcurrent(N,P); 
    arma::mat dtarget(N,P); 
    for ( i =0 ; i< 10000; i++)
    {   
        dcurrent.zeros(); dtarget.zeros();
        getDotAndGradients_chordalfrobenius(current, target, distance, dcurrent, dtarget);
        //getDotAndGradients_binetcauchy(current, target, distance, dcurrent, dtarget);
        //getDotAndGradients_fubinistudy(current, target, distance, dcurrent, dtarget);
        loss = (label - distance)*(label - distance);
        cout << "Loss at iteration " << i << " is " << loss << "\n";
        current += dcurrent*ALPHA*2*(label - distance);
        current = arma::orth(current);
        cout << "CURRENT SUBSPACE:\n" << current;
    }

}

int main()
{
    auto start = high_resolution_clock::now(); 
    arma::mat current(N,P); current.randu();
    arma::mat target(N,P); target.randu();
    target = arma::orth(target);
    current = target;
    cout << "TARGET SUBSPACE:\n" << target ;
    train(current, target);
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    cout << duration.count() << endl; 
    return 0;
}