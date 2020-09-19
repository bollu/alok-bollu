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
using namespace std;
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

void printclosest(arma::Mat<double> target) {
    double bestd[N];
    for (int a = 0; a < N; a++) bestd[a] = 300000.0;
    for (int a = 0; a < N; a++) bestw[a][0] = 0;
    for (int c = 0; c < words; c++) {
        DEBUG_LINE
        const double dist = hack_getDot_binetCauchy(target, c_syn0.slice(c));
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
        printf("Enter 3 words:");
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


        //Get the tangent T_A
        double L = 0.0;
        arma::Mat<double> T_A = log_map(c_syn0.slice(wix[0]), c_syn0.slice(wix[1]), L);
        DEBUG_LINE
        //Transport tangent vector T_A to T_C 
        arma::Mat<double> T_C = parallel(c_syn0.slice(wix[0]), c_syn0.slice(wix[2]), T_A, L);
        DEBUG_LINE
        //Get the new matrix
        arma::Mat<double> target = exp_map(c_syn0.slice(wix[2]), T_C, L);
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
