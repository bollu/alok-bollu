#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <armadillo>
#include <iostream>
#include <cassert>
#include "grad.h"

#define DEBUG_LINE if(0) { printf("%s:%d\n", __FUNCTION__, __LINE__); }
//compute-accuracy /path/to/model.bin < questions-words.txt > output-file.txt
using namespace std;

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

arma::cube c_syn0; 
long long words;
char *vocab;
long long size;
char *bestw[N];
static const long long P = 1;

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



int main(int argc, char **argv)
{
    FILE *f;
    char file_name[max_size]; 
    if (argc < 2) {
        printf("computes analogy for word2vec using word2grass embeddings\n");
        printf("Usage: ./vec-analogy <WORD2VEC-FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
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
                float fl;
                fread(&fl, sizeof(float), 1, f);
                c_syn0(s, p, b) = fl;
            }
        }
        c_syn0.slice(b) = arma::normalise(c_syn0.slice(b));
        // nxp
        printf(" | det %s: %f", vocab + b *max_w, arma::det(c_syn0.slice(b).t()*c_syn0.slice(b)));
        printf("\n");
    }


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
