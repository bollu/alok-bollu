#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <armadillo>
#include <string>
using namespace std;

void getDotAndGradients_chordalfrobenius(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& loss, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    arma::Mat<double> Proj = sub_x*arma::trans(sub_x) - sub_y*arma::trans(sub_y);
    arma::Mat<double> K = Proj*arma::trans(Proj);
    loss = arma::trace(K)/2;
    grad_x = 2*(Proj*sub_x);
    grad_y = -2*(Proj*sub_y);

}

void getDotAndGradients_binetcauchy(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& loss, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    arma::Mat<double> K = arma::trans(sub_x)*sub_y;
    double determinant = arma::det(K);
    arma::Mat<double> K_inv = arma::inv(K);
    loss = 1 - (determinant*determinant);
    grad_x = -2*determinant*determinant*(sub_y*K_inv);
    grad_y = -2*determinant*determinant*(sub_x*K_inv);

}

void getDotAndGradients_fubinistudy(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& loss, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    arma::Mat<double> K1 = arma::trans(sub_x)*sub_y;
    arma::Mat<double> K2 = arma::trans(sub_y)*sub_x;
    arma::Mat<double> K1_inv = arma::inv(K1);
    arma::Mat<double> K2_inv = arma::inv(K2);
    double determinant = arma::det(K1);
    double f = acos(determinant);
    loss = f*f;
    grad_x = 2*f*determinant*(sub_y*K1_inv)/sqrt(1 - (determinant*determinant));
    grad_y = 2*f*determinant*(sub_x*K2_inv)/sqrt(1 - (determinant*determinant));
}

void getDotAndGradients_martin(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& loss, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    arma::Mat<double> K1 = arma::trans(sub_x)*sub_y;
    arma::Mat<double> K2 = arma::trans(sub_y)*sub_x;
    arma::Mat<double> K1_inv = arma::inv(K1);
    arma::Mat<double> K2_inv = arma::inv(K2);
    double determinant = arma::det(K1);
    loss = -2*log(determinant);
    grad_x = -2*sub_y*K1_inv;
    grad_y = -2*sub_x*K2_inv;
}
