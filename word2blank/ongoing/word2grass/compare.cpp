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
const long long int label = 0;
void train(arma::mat current, arma::mat target)
{
    const long long int ndim = current.n_rows;
    const long long int pdim = current.n_cols;

    assert((long long int)target.n_rows == ndim);
    assert((long long int)target.n_cols == pdim);
    long long int i = 0;
    const double ALPHA = 1e-1;
    double distance = 0.0, loss = 0.0;
    arma::mat syn0_gradsq(N,P); syn0_gradsq.zeros();
    arma::mat syn1neg_gradsq(N,P); syn1neg_gradsq.zeros();
    arma::mat dcurrent(N,P); 
    arma::mat dtarget(N,P);
    arma::mat syn0_updates(N,P);
    arma::mat syn1neg_updates(N,P);
    arma::mat clamp_mat(N, P); clamp_mat.fill(1e-8); 
    for ( i =0 ; i< 10000; i++)
    {   
    //    if (i % 10 == 9) { cout << "press key to continue"; getchar(); }
        double syn0_updates_sum = 0;
        double syn1neg_updates_sum = 0;
        dcurrent.zeros(); dtarget.zeros(); syn0_updates.zeros(); syn1neg_updates.zeros();
        getDotAndGradients_chordalfrobenius(current, target, distance, dcurrent, dtarget);
        //getDotAndGradients_binetcauchy(current, target, distance, dcurrent, dtarget);
        //getDotAndGradients_martin(current, target, loss, dcurrent, dtarget);
        //getDotAndGradients_fubinistudy(current, target, distance, dcurrent, dtarget);
        arma::mat temp1 =  -dcurrent*2*(label - distance);
        arma::mat temp2 = -dtarget*2*(label - distance);
        syn0_updates = (temp1*ALPHA)/(arma::sqrt(syn0_gradsq) + clamp_mat);
        syn1neg_updates = (temp2*ALPHA) / (arma::sqrt(syn1neg_gradsq) + clamp_mat);
        syn0_updates_sum = arma::accu(syn0_updates);
        syn1neg_updates_sum = arma::accu(syn1neg_updates);
        //Calculating the matrix r for syn0 and syn1neg which is hadamard product of gradient  
        syn0_gradsq += temp1%temp1; 
        syn1neg_gradsq += temp2%temp2;
        double naturalDist = getNaturalDist(current, target);
        loss = (label - naturalDist); loss *= loss;
        cout << "Chordal Distance " << distance << " (" << label << " - natural[" << naturalDist << "])" <<  " iter " << i << " |" << loss << "|\n";
        if (!isnan(syn0_updates_sum) && !isinf(syn0_updates_sum) && !isnan(syn1neg_updates_sum) && !isinf(syn1neg_updates_sum)) {
            current = arma::orth(current - syn0_updates);
            //target = arma::orth(target - syn1neg_updates);
        }
        // target += dtarget*ALPHA*2*(label - distance); target = arma::orth(target);
        // current = arma::orth(current);
        cout << "CURRENT SUBSPACE:\n" << current;
        //cout << "TARGET SUBSPACE:\n" << target;
    }

}

int main()
{
    auto start = high_resolution_clock::now(); 
    arma::mat current(N,P); current.randu();
    arma::mat target(N,P); target.randu();
    target = arma::orth(target);
    cout << "TARGET SUBSPACE:\n" << target ;
    train(current, target);
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    cout << duration.count() << endl; 
    return 0;
}