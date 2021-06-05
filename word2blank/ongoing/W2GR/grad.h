#include <math.h>
#include <stdio.h>
#include <armadillo>
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

void getDotAndGradients_steifel(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance, 
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

    distance = arma::trace(sub_x.t()*sub_y);
    //arma::Mat<double> K = Proj*arma::trans(Proj);
    //distance = Proj*Proj;
    grad_x = sub_y;
    grad_y = sub_x;

}

void getDot_chordalinner(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance)
{
    const long long int ndim = sub_x.n_rows;
    const long long int pdim = sub_x.n_cols;

    // race condition maybe created because of multi-threading. It's OK, just quit
    // the update this round. because grad_x, grad_y are zeroed, we'll be OK.
    if((long long int)sub_y.n_rows != ndim) {
        printf("ERR\n");
    }
    if((long long int)sub_y.n_cols != pdim) {
        printf("ERR\n"); 
    }
    assert((long long int)sub_y.n_rows == ndim);
    assert((long long int)sub_y.n_cols == pdim);

    double Proj = arma::norm(sub_x.t()*sub_y, "fro");
    //arma::Mat<double> K = Proj*arma::trans(Proj);
    distance = Proj*Proj;
}

arma::Mat<double> getGradients_chordalinner(arma::Mat<double> sub_x, arma::Mat<double> sub_y)
{
    const long long int ndim = sub_x.n_rows;
    const long long int pdim = sub_x.n_cols;

    // race condition maybe created because of multi-threading. It's OK, just quit
    // the update this round. because grad_x, grad_y are zeroed, we'll be OK.
    if((long long int)sub_y.n_rows != ndim) {
        printf("ERR\n"); 
    }
    if((long long int)sub_y.n_cols != pdim) {
        printf("ERR\n"); 
    }
    assert((long long int)sub_y.n_rows == ndim);
    assert((long long int)sub_y.n_cols == pdim);
    
    arma::Mat<double> grad = 2*sub_y*sub_y.t()*sub_x;
    return grad;
}

arma::Mat<double> steif_proj(arma::Mat<double> Z, arma::Mat<double> X)
{
    arma::Mat<double> sym = (X.t()*Z + Z.t()*X)/2;
    arma::Mat<double> proj = Z - X*sym;
    return proj;
}


arma::Mat<double> ortho_proj(arma::Mat<double> grad_x, arma::Mat<double> x)
{
    arma::mat proj = grad_x - x*x.t()*grad_x;
    return proj;
}

void getDotAndGradients_syminner(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance, 
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
    if (!sub_x.is_symmetric(0.01))
        cout << sub_x << endl;
    if (!sub_y.is_symmetric(0.01))
        cout << sub_y << endl;    
    distance = arma::trace(sub_x*sub_y);
    //F_X * X + X * F_X − 2*X*F_X*X
    grad_x = (sub_y*sub_x + sub_x*sub_y) - 2*(sub_x*sub_y*sub_x);
    grad_y = (sub_x*sub_y + sub_y*sub_x) - 2*(sub_y*sub_x*sub_y);
    if (!grad_x.is_symmetric(0.01))
        cout << grad_x << endl;
    if(!grad_y.is_symmetric(0.01))
       cout << grad_y << endl;
}

arma::Mat<double> get_orthomat(arma::Mat<double> P_x)
{
   arma::mat Q; arma::mat R;
   Q = arma::orth(P_x);
   //cout << Q << endl;
   return Q;
}

arma::Mat<double> retraction(arma::Mat<double> X, arma::Mat<double> eta)
{
    const long long int ndim = X.n_rows;
    arma::Mat<double> I(ndim, ndim); I.eye();
    //arma::Mat<double> Y = arma::orth(X);
    arma::Mat<double> Q; arma::Mat<double> R; 
    //arma::Mat<double> inter = (I + eta)*Y;
    arma::Mat<double> inter = I + eta*X - X*eta;
    arma::qr(Q, R, inter);
    arma::mat P = Q*X*Q.t();
    assert(P.is_symmetric(0.01));
    return P;
    //arma::mat P =  Q*Q.t();
    //cout << "new: " << P << endl;
    //assert(P.is_symmetric(0.01));
    //return Q*Q.t(); 
}



void getDotAndGradients_chordalinner(arma::Mat<double> sub_x, arma::Mat<double> sub_y, double& distance, 
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

    double Proj = arma::norm(sub_x.t()*sub_y, "fro");
    distance = Proj*Proj;
    grad_x = 2*sub_y*sub_y.t()*sub_x;
    grad_y = 2*sub_x*sub_x.t()*sub_y;

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

arma::vec getPrincAng(arma::Mat<double> &X, arma::Mat<double> &Y) {

	const long long int n = X.n_rows;
    const long long int p = X.n_cols;
    assert((long long int)Y.n_rows == n);
    assert((long long int)Y.n_cols == p);
    arma::Col<double> s = arma::svd(X.t() * Y);
    s = arma::clamp(s, -1, 1);
    s = arma::acos(s);
    return s;
}


double getChordalDist(arma::Mat<double> &X, arma::Mat<double> &Y) {

	const long long int n = X.n_rows;
    const long long int p = X.n_cols;
    assert((long long int)Y.n_rows == n);
    assert((long long int)Y.n_cols == p);
    arma::Mat<double> Proj = X*X.t() - Y*Y.t();
    return (arma::norm(Proj, "fro")/sqrt(2));
}
