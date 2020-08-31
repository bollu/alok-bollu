#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <armadillo>
#include <string>
using namespace std;

void getDotAndGradients_projection(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    arma::Mat<double> Proj = sub_x*arma::trans(sub_x) - sub_y*arma::trans(sub_y);
    arma::Mat<double> K = Proj*arma::trans(Proj);
    distance = arma::trace(K)/2;
    grad_x = 2*(Proj*sub_x);
    grad_y = -2*(Proj*sub_y);

}


// void getDotAndGradients_geodesic(arma::Mat subspace1, arma::Mat subspace2, double &dot, 
// arma::Mat &grad1, arma::Mat &grad2)
// {   


// }