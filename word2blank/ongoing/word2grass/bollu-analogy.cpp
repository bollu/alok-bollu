// https://manoptjl.org/stable/
// https://arxiv.org/pdf/physics/9806030.pdf
// http://svcl.ucsd.edu/publications/journal/2016/ggr/supplementary_material.pdf <- best reference for equations!!!
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <armadillo>
#include <vector>
#include <string>
#include "grad.h"
#include <iostream>

#define DEBUG_LINE if(0) { printf("%s:%d\n", __FUNCTION__, __LINE__); }

// optimization on steifel manifold: http://noodle.med.yale.edu/~hdtag/notes/steifel_notes.pdf

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

arma::cube c_syn0; 
long long words;
char *vocab;
long long P, size;
char *bestw[N];

// I don't understand what this does.
double __attribute__((alwaysinline)) hack_getDot_binetCauchy(const
        arma::Mat<double> &sub_x, const arma::Mat<double> &sub_y) {
    arma::Mat<double> XtY = arma::trans(sub_x)*sub_y;
    double determinant_xty = arma::det(XtY);
    arma::Mat<double> xty_inv = arma::inv(XtY);

    arma::Mat<double> YtX = arma::trans(sub_y)*sub_x;
    double determinant_ytx = arma::det(YtX);
    arma::Mat<double> ytx_inv = arma::inv(YtX);

    return 1 - (determinant_xty*determinant_xty);

}

// http://svcl.ucsd.edu/publications/journal/2016/ggr/supplementary_material.pdf
arma::Mat<double> log_paper(const arma::Mat<double> start, const arma::Mat<double> end) {
	// TODO: does '0' / 'econ' in SVD matter?
    // function H = log map(X,Y)
    // % LOG MAP computes the inverse exponential map on the Grassmannian.
    // %
    // % Input:
    // % X − (n x p) matrix (representer for subspace \mathcal{X})
    // % Y − (n x p) matrix (representer for subspace \mathcal{Y})
    // %
    // % Output:
    // %
    // % H − (n x p) direction matrix in the tangent space at subspace \mathcal{X}
    // % H = log_x(y) <=> y = exp_x(h)
    // ytx = Y.'*X;
	// At = Y.'−ytx*X.';
	// Bt = ytx\At;
	// [U, S, V] = svd(Bt.', 'econ');
	// U = U(:, 1:p);
	// S = diag(S);
	// S = S(1:p);
	// V = V(:, 1:p);
	// H = U*diag(atan(S))*V.';

    DEBUG_LINE
	assert(start.n_rows == end.n_rows);
	const int p = start.n_cols; assert(p == end.n_cols);

	// [unused, p] = size(X);
	arma::Mat<double> ytx = end.t() * start;
	arma::Mat<double> At = end.t()  - ytx * start.t();
	arma::Mat<double> Bt = arma::solve(ytx, At);
	arma::Mat<double> U, V;
	arma::Col<double> s;
    DEBUG_LINE
    arma::svd_econ(U, s, V, Bt.t());

    DEBUG_LINE
    U = U.cols(0, p-1);
    DEBUG_LINE
    V = V.cols(0, p-1);
    DEBUG_LINE
	s.resize(p);
    DEBUG_LINE
    DEBUG_LINE
    return U * arma::diagmat(arma::atan(s)) * V.t();
    // arma::Mat<double> S = arma::diag(S); S = S.col
     
  
}

// http://svcl.ucsd.edu/publications/journal/2016/ggr/supplementary_material.pdf <- best reference for equations!!!
// 2.5.1: Geodesics (Grassmanian) http://www-math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf
arma::Mat<double> exp_paper(const arma::Mat<double> start, const arma::Mat<double> tgt, double t = 1.0) {
  
	// TODO: does '0' / 'econ' in SVD matter?
    // % EXP MAP computes the exponential map on the Grassmannian G(p,n).
    // %
    // % Input:
    // % Y − (n x p) matrix (representer for subspace \mathcal{Y})
    // % D − (n x p) matrix (tangent vector at \mathcal{Y})
    // % t − Time in [0,1]
    // %
    // % Output:
    // %
    // % Z − (n x p) matrix; Result of walking along the geodesic, defined
    // % by (Y,D), for time t;
    // %
    // % Reference − Eq. (2.65) from
    // %
    // % Edelman et al.
    // % The Geometry of algorithms with orthogonality constraints
    // % Online: http://arxiv.org/pdf/physics/9806030.pdf

	// tD = t*D;
	// [U, S, V] = svd( tD, 0 );
	// cosS = diag( cos( diag( S ) ) );
	// sinS = diag( sin( diag( S ) ) );
	// Z = Y*V*cosS*V' + U*sinS*V';
	// [Q, unused] = qr( Z, 0 );
	// Z = Q;

	arma::Mat<double> U, V; arma::Col<double> s;
    DEBUG_LINE
	arma::svd_econ(U, s, V, tgt);
    DEBUG_LINE
    s *= t;
    arma::Mat<double> cosS = arma::diagmat(arma::cos(s));
    arma::Mat<double> sinS = arma::diagmat(arma::sin(s));
    DEBUG_LINE
    arma::Mat<double> Z = start * V * cosS * V.t() + U * sinS * V.t();
    DEBUG_LINE
    arma::Mat<double> Q, unused;
    DEBUG_LINE
    arma::qr_econ(Q, unused, Z);
    return Q;
}


// if if tgtStart ∈ TangentSpace(start),
//    move tangent vector  tgtStart to TangentSpace(end)
// eqn 2.5.2: http://www-math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf
arma::Mat<double> parallel_paper(const arma::Mat<double> start, const arma::Mat<double> end, const arma::Mat<double> tgtStart,
        double t = 1) {
    // function Delta = parallel transport(O1,B,O2,t)
    // % PARALLEL TRANSPORT parallel transports a tangent along geodesic.
    // %
    // % Input:
    // % O1 − (n x p) matrix (representer for subspace \mathcal{X})
    // % O2 − (n x p) matrix (representer for subspace \mathcal{Y})
    // % B − (n x p) matrix (tangent vector at \mathcal{X})
    // % t − time in [0,1]
    // %
    // % Output:
    // %
    // % Delta − (n x p) matrix (tangent vector B, transported to the
    // % tangent space at \mathcal{Y}
    // H = log map(O1,O2); % compute direction matrix
    // [U,S,V] = svd(H, 'econ');
    // Sigmat = diag(S)*t;
    // M = [diag(−sin(Sigmat)); diag(cos(Sigmat))];
    // part0 =[O1*V U]*(M*(U'*B));
    // part1 = B − U*(U'*B);
    // Delta = part0 + part1;
    // end

    DEBUG_LINE
    const arma::Mat<double> start2End = log_paper(start, end);
    DEBUG_LINE
	arma::Mat<double> U, V; arma::Col<double> s;
    DEBUG_LINE
	arma::svd_econ(U, s, V, start2End);
    DEBUG_LINE
    s *= t;
    DEBUG_LINE
    arma::Mat<double> M = arma::join_vert(-arma::diagmat(arma::sin(s)), arma::diagmat(arma::cos(s)));
    DEBUG_LINE
    arma::Mat<double> part0 = arma::join_horiz(start * V, U)* M * U.t() * tgtStart;
    DEBUG_LINE
    arma::Mat<double> part1 = tgtStart - U * (U.t() * tgtStart);
    DEBUG_LINE
    return part0 + part1;
}

void printclosest(arma::Mat<double> target) {
    double bestd[N];
    for (int a = 0; a < N; a++) bestd[a] = 300000.0;
    for (int a = 0; a < N; a++) bestw[a][0] = 0;
    for (int c = 0; c < words; c++) {
        DEBUG_LINE
        const double dist = getNaturalDist(target, c_syn0.slice(c));
        DEBUG_LINE
        for (int a = 0; a < N; a++) {
            if (dist < bestd[a]) {
                for (int d = N -1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], &vocab[c * max_w]);
                break;
            }
        }
    }
    for (int a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
}

int main(int argc, char **argv) {
    FILE *f;
    char file_name[max_size]; 
    if (argc < 2) {
        printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
        return 0;
    }
    strcpy(file_name, argv[1]);
    f = fopen(file_name, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    fscanf(f, "%lld", &P);
    printf("words: %lld | size: %lld | P: %lld\n", words, size, P);

    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (int a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    c_syn0.set_size(size, P, words);

    for (int b = 0; b < words; b++) {
        printf("PROGRESS: %d/%d: %f", b, words, (b * 100.0) / words);
        int a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        vocab[b * max_w + a] = 0;
        for(int p  = 0; p < P; p++) {
            for (int s = 0; s < size; s++) {
                fread(&c_syn0(s, p, b), sizeof(double), 1, f);
            }
        }
        // nxp
        printf(" | det %s: %f", vocab + b *max_w, arma::det(c_syn0.slice(b).t()*c_syn0.slice(b)));
        printf("\n");
    }

    fclose(f);

    while(1) {
        printf("Enter  3 words:");
        char word[3][512];
        scanf("%s %s %s", word[0], word[1], word[2]);
        printf("computing |%s:%s::%s:?|\n", word[0], word[1], word[2]);
        if (!strcmp(word[0], "EXIT")) { return 0; }

        int wix[3] = {-1, -1, -1};
        for(int i = 0; i < words; i++) {
            for(int w = 0; w < 3; ++w) {
                if(!strcmp(vocab + i *max_w, word[w])) { wix[w] = i; }
            }
        }

        for(int w = 0; w < 3; ++w) {
            if (wix[w] == -1) { 
                printf("|%s| out of vocabulary.\n", word[w]);
            }
        }

        if (wix[0] == -1 || wix[1] == -1 || wix[2] == -1) { continue; }


        arma::Mat<double> tgt0to1_at_0 = log_paper(c_syn0.slice(wix[0]), c_syn0.slice(wix[1]));
        DEBUG_LINE
        arma::Mat<double> tgt0to1_at_2 = parallel_paper(c_syn0.slice(wix[0]), c_syn0.slice(wix[2]), tgt0to1_at_0);
        DEBUG_LINE
        arma::Mat<double> target = exp_paper(c_syn0.slice(wix[2]), tgt0to1_at_2);
        // arma::Mat<double> target = exp(c_syn0.slice(wix[2]), tgt0to1_at_0);
        DEBUG_LINE
        printclosest(target);

        /*
        const int NSTEPS = 10;
        for(int i = 0; i <= NSTEPS; ++i) {
            printf("geodesic (%4.2f)\n", float(i)/NSTEPS*100.0);
            printclosest(exp(c_syn0.slice(wix[0]), tgt0to1_at_0 * float(i)/NSTEPS));
        }
        */

    }
    return 0;
}
