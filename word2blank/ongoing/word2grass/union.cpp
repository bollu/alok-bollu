#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <armadillo>
#include <iostream>
#include "grad.h"

using namespace std;
const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

arma::cube c_syn0; 
long long words;
double samples=1.0;
char *vocab;
long long P, size;
char *bestw[N];



arma::Mat<double> subspace(arma::Mat<double> U, arma::Mat<double> V, 
    arma::Col<double> theta, arma::Mat<double> start, double t) {

    arma::Mat<double> end = start*V*arma::diagmat(arma::cos(theta*t))*V.t() + U*arma::diagmat(arma::sin(theta*t))*V.t();
    return end;
}


void printclosest(arma::Mat<double> target, int ind) {
    double best;
    best = 300000.0;
    bestw[ind][0] = 0;
    for (int c = 0; c < words; c++) {
    double dist = getNaturalDist(target, c_syn0.slice(c));
    if (dist < best) {
            best = dist;
            strcpy(bestw[ind], &vocab[c * max_w]);
        }
    }
    printf("%50s\t\t%f\n", bestw[ind], best);
}


int main(int argc, char **argv) {
    FILE *f;
    char file_name[max_size]; 
    if (argc < 2) {
        printf("Usage: ./union <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
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
        printf("Enter 2 words:");
        char word[2][512];
        scanf("%s %s", word[0], word[1]);
        printf("computing |%s U %s|\n", word[0], word[1]);
        if (!strcmp(word[0], "EXIT")) { return 0; }

        int wix[2] = {-1, -1};
        for(int i = 0; i < words; i++) {
            for(int w = 0; w < 2; ++w) {
                if(!strcmp(vocab + i *max_w, word[w])) { wix[w] = i; }
            }
        }

        for(int w = 0; w < 2; ++w) {
            if (wix[w] == -1) { 
                printf("|%s| out of vocabulary.\n", word[w]);
            }
        }

        if (wix[0] == -1 || wix[1] == -1 ) { continue; }
        arma::Mat<double> start = c_syn0.slice(wix[0]);
        arma::Mat<double> end = c_syn0.slice(wix[1]);
	    const long long int n = start.n_rows;
        const long long int p = start.n_cols;
        assert((long long int)end.n_rows == n);
        assert((long long int)end.n_cols == p);
        arma::Mat<double> I(n,n); I.eye();
        arma::Mat<double> PI_K = I - (start*arma::trans(start));
        arma::Mat<double> K = end*arma::inv(arma::trans(start)*end);
        arma::Mat<double> G = PI_K*K;
        arma::Mat<double> U, V; arma::Col<double> s;
        arma::svd_econ(U, s, V, G);
        arma::Col<double> theta = arma::atan(s);
        double iter = 0.0;
        while(iter <= samples)
        {
            arma::Mat<double> subsp = subspace(U, V, theta, start, iter);
            printclosest(subsp, iter);
            iter += 0.005;
        }    
    }   
    return 0;
}