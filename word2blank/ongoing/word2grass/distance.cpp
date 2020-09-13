#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <armadillo>
#include <vector>
#include <string>
#include "grad.h"

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

arma::cube c_syn0; 

double __attribute__((alwaysinline)) hack_getDot_binetCauchy(const
        arma::Mat<double> &sub_x, const arma::Mat<double> &sub_y) {
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

    return 1 - (determinant_xty*determinant_xty);

}

int main(int argc, char **argv) {
    FILE *f;
    char *bestw[N];
    char file_name[max_size], st[100][max_size];
    double dist,  bestd[N];
    long long words, size, a, b, c, d, P;
    char *vocab;
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
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    c_syn0.set_size(size, P, words);

    for (b = 0; b < words; b++) {
        printf("\nPROGRESS:\n");
        printf("\r%d/%d: %f", b, words, (b * 100.0) / words);
        a = 0;
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
    }

    fclose(f);

    while(1) {
        for (a = 0; a < N; a++) bestd[a] = -1;
        for (a = 0; a < N; a++) bestw[a][0] = 0;

        printf("Enter word or sentence (EXIT to break): ");
        char word[512];
        scanf("%s", word);
        if (!strcmp(word, "EXIT")) { return 0; }

        int wix = -1;
        for(int i = 0; i < words; i++) {
            if(!strcmp(vocab + i *max_w, word)) { wix = i; break; }
        }

        if (wix == -1) { printf("|%s| out of vocabulary.\n", word); }

        for (a = 0; a < N; a++) bestd[a] = 300000.0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        for (c = 0; c < words; c++) {
            dist =  hack_getDot_binetCauchy(c_syn0.slice(wix), c_syn0.slice(c));
            for (a = 0; a < N; a++) {
                if (dist < bestd[a]) {
                    for (d = N -1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    }
    return 0;
}
