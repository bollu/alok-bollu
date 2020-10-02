#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <assert.h>
#include <armadillo>
#include <vector>
#include <string>
#include "grad.h"

#define MAX_EXP 
using namespace std;

int SIZE  = 3;
int NVEC = 4;
int NITER = 1500;
int NEG = NVEC - 1;
double ALPHA = 1e-3;


double sigmoid(double x)
{
    if (isinf(exp(x))) return 1;
    return exp(x)/(1 + exp(x));
}

void generate_vectors(arma::mat& focus, arma::mat& context)
{
    printf("INITIALISING FOCUS VECTORS\n");
    arma::arma_rng::set_seed_random();
    for (int i=0; i<NVEC; i++) focus.col(i) = arma::randn<arma::vec>(SIZE);
    printf("done\n");
    printf("INITIALISING CONTEXT VECTORS\n");
    arma::arma_rng::set_seed_random();
    //for (int i=0; i<NVEC; i++) context.col(i) = arma::randn<arma::vec>(SIZE);
    context.zeros();
    printf("done\n"); 
}

int main()
{   
    int label;
    arma::mat focus(SIZE, NVEC);
    arma::mat context(SIZE, NVEC);
    arma::mat focus_gradsq(SIZE, NVEC); focus_gradsq.zeros();
    arma::mat context_gradsq(SIZE, NVEC); context_gradsq.zeros(); 
    arma::vec clamp_vec(SIZE); clamp_vec.fill(1e-8);
    generate_vectors(focus, context);
    for (int i=0; i<NITER; i++)
    {
        for (int j=0; j<NVEC; j++)
        {   
            double focus_updates_sum = 0;
            double context_updates_sum = 0;
            arma::vec buff0 = focus.col(j);
            for(int k=0; k<NEG; k++)
            {
                if (k == j) label = 1; else label = 0;
                cout << "|focus vector| " << focus.col(j).t() ;
                cout << "|context vector| " << context.col(k).t();
                double dot = arma::dot(focus.col(j), context.col(k));
                cout << "|iter|: " << i << " |j|: " << j << " |k|: " << k << " |label|: " << label << " |dot|: " << dot << endl;
                cout << "|sigmoid(dot)|: " << sigmoid(dot) << " |label -sigmoid(dot)|: " << (label - sigmoid(dot)) << endl;
                //gradient calculation for focus and context
                arma::vec temp1 = -2*(label - sigmoid(dot))*sigmoid(dot)*(1 - sigmoid(dot))*context.col(k);
                arma::vec temp2 = -2*(label - sigmoid(dot))*sigmoid(dot)*(1 - sigmoid(dot))*focus.col(i);
                cout << "|focus gradient|" << temp1.t() ;
                cout << "|context gradient|" << temp2.t() ;
                //calculates the update values for focus vector 
                arma::vec focus_updates = (temp1*ALPHA)/(arma::sqrt(focus_gradsq.col(j)) + clamp_vec);
                cout << "|FOCUS UPDATES| " << focus_updates.t();
                arma::vec context_updates = (temp2*ALPHA)/(arma::sqrt(context_gradsq.col(k)) + clamp_vec);
                cout << "|CONTEXT UPDATES| " << context_updates.t();
                focus_updates_sum = arma::accu(focus_updates);
                context_updates_sum = arma::accu(context_updates);
                //store the sum of gradient squares
                focus_gradsq.col(j) += temp1%temp1; //
                context_gradsq.col(k) += temp2%temp2;
                if (!isnan(focus_updates_sum) && !isinf(focus_updates_sum) && !isnan(context_updates_sum) && !isinf(context_updates_sum)) {
                    buff0 -= focus_updates;
                    context.col(k) -= context_updates;
                }
                // buff0 -= temp1*ALPHA;
                // context.col(k) -= temp2*ALPHA;
            }
            focus.col(j) = buff0;
        }
    }
    for (int i=0 ; i<NVEC; i++)
        cout << focus.col(i) << endl;
    return 0;
}