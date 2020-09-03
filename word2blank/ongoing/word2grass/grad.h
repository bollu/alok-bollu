#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <armadillo>
#include <string>
using namespace std;

void getDotAndGradients_chordalfrobenius(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    arma::Mat<double> Proj = sub_x*arma::trans(sub_x) - sub_y*arma::trans(sub_y);
    arma::Mat<double> K = Proj*arma::trans(Proj);
    distance = sqrt(arma::trace(K)/2);
    grad_x = (Proj*sub_x)/distance;
    grad_y = -(Proj*sub_y)/distance;

}

void getDotAndGradients_binetcauchy(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    const long long int ndim = sub_x.n_rows;
    const long long int pdim = sub_x.n_cols;

    assert((long long int)sub_y.n_rows == ndim);
    assert((long long int)sub_y.n_cols == pdim);

    arma::Mat<double> XtY = arma::trans(sub_x)*sub_y;
    double determinant_xty = arma::det(XtY);
    arma::Mat<double> xty_inv = arma::inv(XtY);

    arma::Mat<double> YtX = arma::trans(sub_y)*sub_x;
    double determinant_ytx = arma::det(YtX);
    arma::Mat<double> ytx_inv = arma::inv(YtX);

    distance = 1 - (determinant_xty*determinant_xty);
    grad_x = -2*determinant_xty*determinant_xty*(sub_y*xty_inv);
    grad_y = -2*determinant_ytx*determinant_ytx*(sub_x*ytx_inv);

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
