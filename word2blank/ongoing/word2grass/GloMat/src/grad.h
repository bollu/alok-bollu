#ifndef GRAD_H
#define GRAD_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>

using namespace std;

#define DEBUG_LINE if(0) { printf("%s:%d\n", __FUNCTION__, __LINE__); }

void getDotAndGradients_chordalfrobenius(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    const long long int ndim = sub_x.n_rows;
    const long long int pdim = sub_x.n_cols;

    // race condition maybe created because of multi-threading. It's OK, just quit
    // the update this round. because grad_x, grad_y are zeroed, we'll be OK.
    if((long long int)sub_y.n_rows != ndim) {
        printf("ERR\n"); return;
    }
    if((long long int)sub_y.n_cols != pdim) {
        printf("ERR\n"); return;
    }
    assert((long long int)sub_y.n_rows == ndim);
    assert((long long int)sub_y.n_cols == pdim);

    arma::Mat<double> Proj = sub_x*arma::trans(sub_x) - sub_y*arma::trans(sub_y);
    //arma::Mat<double> K = Proj*arma::trans(Proj);
    distance = arma::norm(Proj, "fro")/sqrt(2);
    grad_x = (Proj*sub_x)/distance;
    grad_y = -(Proj*sub_y)/distance;

}

void getDotAndGradients_binetcauchy(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    const long long int ndim = sub_x.n_rows;
    const long long int pdim = sub_x.n_cols;

    // race condition maybe created because of multi-threading. It's OK, just quit
    // the update this round. because grad_x, grad_y are zeroed, we'll be OK.
    if((long long int)sub_y.n_rows != ndim) {
        printf("ERR\n"); return;
    }
    if((long long int)sub_y.n_cols != pdim) {
        printf("ERR\n"); return;
    }
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

void getDotAndGradients_fubinistudy(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance, 
arma::Mat<double>& grad_x, arma::Mat<double>& grad_y)
{
    double sign_det2 , sign_det1;
    arma::Mat<double> K1 = arma::trans(sub_x)*sub_y;
    arma::Mat<double> K2 = arma::trans(sub_y)*sub_x;
    arma::Mat<double> K1_inv = arma::inv(K1);
    arma::Mat<double> K2_inv = arma::inv(K2);
    double determinant1 = arma::det(K1);
    double determinant2 = arma::det(K2);
    double abs_determinant1 = abs(determinant1);
    double abs_determinant2 = abs(determinant2);
    distance = acos(abs_determinant1);
    if (determinant1 < 0) 
        sign_det1 = -1; 
    else 
        sign_det1 = 1;
    if (determinant2 < 0)
        sign_det2 = -1;
    else
        sign_det2 = 1; 
    grad_x = -1*sign_det1*determinant1*(sub_y*K1_inv)/sqrt(1 - (abs_determinant1*abs_determinant1));
    grad_y = -1*sign_det2*determinant2*(sub_x*K2_inv)/sqrt(1 - (abs_determinant2*abs_determinant2));
}


arma::Mat<double> log_map(const arma::Mat<double> start, const arma::Mat<double> end, double &L) 
{
    DEBUG_LINE
	const long long int n = start.n_rows;
    const long long int p = start.n_cols;
    DEBUG_LINE
    assert((long long int)end.n_rows == n);
    assert((long long int)end.n_cols == p);
    DEBUG_LINE
    arma::Mat<double> I(n,n); I.eye();
    DEBUG_LINE
    arma::Mat<double> PI_K = I - (start*arma::trans(start));
	DEBUG_LINE
    arma::Mat<double> K = end*arma::inv(arma::trans(start)*end);
    DEBUG_LINE
    arma::Mat<double> G = PI_K*K;
    arma::Mat<double> U, V; arma::vec s;
    DEBUG_LINE
	arma::svd_econ(U, s, V, G);
    DEBUG_LINE
    arma::Col<double> theta = arma::atan(s);
    DEBUG_LINE
    L = sqrt(arma::accu(theta % theta));
    DEBUG_LINE
    arma::Mat<double> T_A = U * arma::diagmat(theta) * V.t();
    return T_A;  
}


arma::Mat<double> parallel(const arma::Mat<double> start, const arma::Mat<double> end, const arma::Mat<double> tgtStart, double L) 
{
    DEBUG_LINE
	const long long int n = start.n_rows;
    const long long int p = start.n_cols;
    DEBUG_LINE
    assert((long long int)end.n_rows == n);
    assert((long long int)end.n_cols == p);
    DEBUG_LINE
    arma::Mat<double> I(n,n); I.eye();
    DEBUG_LINE
    arma::Mat<double> PI_K = I - (start*arma::trans(start));
	DEBUG_LINE
    arma::Mat<double> K = end*arma::inv(arma::trans(start)*end);
    DEBUG_LINE
    arma::Mat<double> G = PI_K*K;
    arma::Mat<double> U, V; arma::vec s;
    DEBUG_LINE
	arma::svd_econ(U, s, V, G);
    DEBUG_LINE
    arma::Col<double> theta = arma::atan(s);
    DEBUG_LINE
    arma::Mat<double> tgt_move = (-start*V*arma::diagmat(arma::sin(theta))*U.t()*tgtStart) + (U*arma::diagmat(arma::cos(theta))*U.t()*tgtStart) + ((I - (U*U.t()))*tgtStart);
    DEBUG_LINE
    //arma::Mat<double> tgt_end =  tgt_move*tgtStart;
    DEBUG_LINE
    return tgt_move;
}

arma::Mat<double> exp_map(const arma::Mat<double> start, const arma::Mat<double> tgt, double L) 
{
    arma::Mat<double> U, V; arma::vec s;
    DEBUG_LINE
    //arma::Mat<double> act_tgt = tgt;//*L;
	arma::svd_econ(U, s, V, tgt);
    DEBUG_LINE
    arma::Mat<double> end = start*V*arma::diagmat(arma::cos(s))*V.t() + U*arma::diagmat(arma::sin(s))*V.t();
    DEBUG_LINE
    //end = arma::orth(end);
    DEBUG_LINE
    return end;
}

double getNaturalDist(arma::Mat<double> &X, arma::Mat<double> &Y) {

	const long long int n = X.n_rows;
    const long long int p = X.n_cols;
    assert((long long int)Y.n_rows == n);
    assert((long long int)Y.n_cols == p);
    arma::Col<double> s = arma::svd(X.t() * Y);
    s = arma::clamp(s, -1, 1);
    s = arma::acos(s);
    return sqrt(arma::accu(s % s));
}


double getChordalDist(arma::Mat<double> &X, arma::Mat<double> &Y) {

	const long long int n = X.n_rows;
    const long long int p = X.n_cols;
    assert((long long int)Y.n_rows == n);
    assert((long long int)Y.n_cols == p);
    arma::Mat<double> Proj = X*X.t() - Y*Y.t();
    return (arma::norm(Proj, "fro")/sqrt(2));
}


#endif /* GRAD_H */