#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <assert.h>
#include <armadillo>
#include <vector>
#include <string>
#include "grad.h"

using namespace std;

int N  = 3;
int P = 2;
int NMAT = 4;
long long int NITER = 90000;
int NEG = NMAT - 1;
double ALPHA = 2e-2;


double sigmoid(double x)
{
    if (isinf(exp(x))) return 1;
    return exp(x)/(1 + exp(x));
}

void generate_matrices(arma::cube& focus, arma::cube& context)
{
    printf("INITIALISING FOCUS MATRICES\n");
    arma::arma_rng::set_seed_random();
    printf("INITIALISING CONTEXT MATRICES\n");
    arma::arma_rng::set_seed_random();
    for (int i=0; i<NMAT; i++) {
        focus.slice(i) = arma::orth(arma::randn<arma::mat>(N, P));
        arma::uword r = arma::rank(focus.slice(i));
        if ((long long)r != P) printf("Gotcha\n");
        context.slice(i) = arma::orth(arma::randn<arma::mat>(N, P));
        arma::uword r1 = arma::rank(context.slice(i));
        if ((long long)r1 != P) printf("Gotcha\n");
    }
    printf("BEFORE\n");
    for (int i=0 ; i<NMAT; i++)
    {
        for (int j=0; j<NMAT; j++)
        {
            //arma::vec angles = getPrincAng(focus.slice(i), context.slice(j));
            double dist = getNaturalDist(focus.slice(i), context.slice(j));
            //double dist = getChordalDist(focus.slice(i), context.slice(j));
            cout << "Focus vector No: " << i << " ; Context Vector No: " << j << "; distance: " << dist << endl;
        }
    }
    // printf("BEFORE CONTEXT\n");
    // for (int i=0 ; i<NMAT; i++)
    //     cout << context.slice(i) << endl;
}

int main()
{   
    int label;
    arma::cube focus(N, P, NMAT);
    arma::cube context(N, P, NMAT);
    arma::cube focus_gradsq(N, P, NMAT); focus_gradsq.zeros();
    arma::cube context_gradsq(N, P, NMAT); context_gradsq.zeros(); 
    arma::mat clamp_mat(N, P); clamp_mat.fill(1e-8);
    arma::mat grad_context(N, P);
    arma::mat grad_focus(N, P);
    generate_matrices(focus, context);
    for (long long int i=0; i<NITER; i++)
    {
        for (long long int j=0; j<NMAT; j++)
        {   
            double focus_updates_sum = 0;
            double context_updates_sum = 0;
            arma::mat buff0 = focus.slice(j);
            for(int k=0; k<NMAT; k++)
            {
                if (k == j) label = 0; else label = 1;
                //cout << "|focus vector| " << focus.slice(j).t() ;
                //cout << "|context vector| " << context.slice(k).t();
                double f = 0.0;
                grad_focus.zeros(); grad_context.zeros();
                getDotAndGradients_chordalfrobenius(focus.slice(j), context.slice(k), f, grad_focus, grad_context);
                //cout << "|iter|: " << i << " |j|: " << j << " |k|: " << k << " |label|: " << label << " |dist|: " << f << endl;
                //cout << "|sigmoid(dot)|: " << sigmoid(dot) << " |label -sigmoid(dot)|: " << (label - sigmoid(dot)) << endl;
                //gradient calculation for focus and context
                //arma::mat temp1 = -2*(label - sigmoid(f))*sigmoid(f)*(1 - sigmoid(f))*grad_focus;
                //arma::mat temp2 = -2*(label - sigmoid(f))*sigmoid(f)*(1 - sigmoid(f))*grad_context;
                arma::mat temp1 = (-2*(label - f/sqrt(P))*grad_focus)/sqrt(P);
                arma::mat temp2 = (-2*(label - f/sqrt(P))*grad_context)/sqrt(P);
                //cout << "|focus gradient|" << temp1.t() ;
                //cout << "|context gradient|" << temp2.t() ;
                //calculates the update values for focus vector 
                arma::mat focus_updates = (temp1*ALPHA)/(arma::sqrt(focus_gradsq.slice(j)) + clamp_mat);
                //cout << "|FOCUS UPDATES| " << focus_updates.t();
                arma::mat context_updates = (temp2*ALPHA)/(arma::sqrt(context_gradsq.slice(k)) + clamp_mat);
                //cout << "|CONTEXT UPDATES| " << context_updates.t();
                focus_updates_sum = arma::accu(focus_updates);
                context_updates_sum = arma::accu(context_updates);
                //store the sum of gradient squares
                focus_gradsq.slice(j) += temp1%temp1; //
                context_gradsq.slice(k) += temp2%temp2;
                if (!isnan(focus_updates_sum) && !isinf(focus_updates_sum) && !isnan(context_updates_sum) && !isinf(context_updates_sum)) {
                    buff0 -= focus_updates;
                    context.slice(k) = arma::orth(context.slice(k) - context_updates);
                }
                // buff0 -= temp1*ALPHA;
                // context.col(k) -= temp2*ALPHA;
            }
            focus.slice(j) = arma::orth(buff0);
        }
    }
    printf("AFTER\n");
    for (int i=0 ; i<NMAT; i++)
    {
        for (int j=0; j<NMAT; j++)
        {
            //arma::vec angles = getPrincAng(focus.slice(i), context.slice(j));
            double dist = getNaturalDist(focus.slice(i), context.slice(j));
            //double dist = getChordalDist(focus.slice(i), context.slice(j));
            cout << "Focus vector No: " << i << " ; Context Vector No: " << j << "; distance: " << dist << endl;
        }
    }
    return 0;
}