#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <armadillo>
#include "grad.h"

using namespace std; 

#define max_size 2000
#define N 25
#define max_w 100

FILE *f;
char st1[max_size];
char *bestw[N];
char file_name[max_size], st[100][max_size];
double dist, len, bestd[N], vec[max_size];
long long words, P, size, row, col, a, b, c, d, cn, bi[100];
double *M;
char *vocab;

double sim(int w1, int w2) {
    assert (w1 >= 0);
    assert (w2 >= 0);
    arma::mat M1(P, size); M1.zeros();
    arma::mat M2(P, size); M2.zeros();
    for ( row = 0; row < P; row++) for ( col = 0; col < size; col++) M1(row,col) = M[(size*P*w1) + (row*size) + col]; 
    for ( row = 0; row < P; row++) for ( col = 0; col < size; col++) M2(row,col) = M[(size*P*w2) + (row*size) + col]; 

    // double distance = arma::trace(M1*M2.t());
    // return distance;
    // return arma::trace(M1*M2.t());
    double Proj = arma::norm(M1*M2.t(),"fro");
    return Proj*Proj;
}


int main(int argc, char **argv) {

    printf("are you sure you want to run this? You likely wish to run spearman.py");
    if (argc < 3) {
        printf(
            "Usage: ./simlex-accuracy <VECFILE> <SIMLEXFILE> [OUTFILE]"
            "\nwhere VECFILE contains word projections in the BINARY FORMAT"
            "\nSIMLEXFILE is the SimLex-999.txt from SimLex"
            "\n[OUTFILE] is the optional file to dump <simlexscore>:<vecscore>");
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
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (double *)malloc((long long)words * (long long )P* (long long)size * sizeof(double));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)words * size * sizeof(double) / 1048576, words, size);
        return -1;
    }
    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        vocab[b * max_w + a] = 0;
        for (row = 0; row < P; row++) for (col = 0; col < size; col++) fread(&M[col + (row * size) + (b * size * P)], sizeof(double), 1, f);
    }
    fclose(f);

    f = fopen(argv[2], "r");

    // throw away first line.
    {
        char *throwaway; size_t throwi;
        getline(&throwaway, &throwi, f);
    }

    static const int MAX_LINES_SIMLEX = 1002;
    double *simlexes = (double *)malloc(sizeof(double) * MAX_LINES_SIMLEX);
    double *oursims = (double *)malloc(sizeof(double) * MAX_LINES_SIMLEX);

    char word1[max_size], word2[max_size], word3[max_size];
    int n = 0;
    for(; !feof(f);) {
        // word1\tword2\tPOS[1letter]\tSimLex999\t
        char *linebuf = 0;
        size_t linelen;
        getline(&linebuf, &linelen, f);
        if (strlen(linebuf) == 0) { break; }
        int i = 0;
        while(linebuf[i] != '\t' && linebuf[i] != ' ') {
            word1[i] = linebuf[i]; i++;
        }
        const int w2_startix = i+1;
        int j = 0;
        word1[i] = '\0';
        while(linebuf[w2_startix + j] != '\t' && 
                linebuf[w2_startix +j] != ' ') {
            word2[j] = linebuf[w2_startix + j]; j++;
        }
        word2[j] = '\0';


        // skip \tPOS\t
        assert (linebuf[w2_startix + j] == '\t' || linebuf[w2_startix + j] == ' ');

        assert (linebuf[w2_startix + j+1] == 'A' ||
                linebuf[w2_startix + j+1] == 'N' ||
                linebuf[w2_startix + j+1] == 'V');

        assert (linebuf[w2_startix + j+2] == '\t' || linebuf[w2_startix + j+2] == ' ');
        const int w3_startix = w2_startix + j + 3;
        int k = 0;
        while(linebuf[w3_startix + k] != '\t' && 
              linebuf[w3_startix + k] != ' ') {
            word3[k] = linebuf[w3_startix + k]; k++;
        }
        word3[k] = '\0';

        simlexes[n] = atof(word3);


        // skip word and grab simlex score;
        fprintf(stderr, "> |%s| :: |%s| : simlex(%f)  <\n", word1, word2, simlexes[n]);
        free(linebuf);

        int w1ix = -1, w2ix = -1;
        for(int i = 0; i < words; ++i) {
            if (!strcmp(&vocab[max_w*i], word1)) {
                w1ix = i;
                fprintf(stderr, "\tvocab[%d] = %s\n", w1ix, &vocab[max_w*w1ix]);
            }
            if (!strcmp(&vocab[max_w*i], word2)) {
                w2ix = i;
                fprintf(stderr, "\tvocab[%d] = %s\n", w2ix, &vocab[max_w*w2ix]);
            }
        }

        if (w1ix == -1 || w2ix == -1) {
            fprintf(stderr, "\tSKIPPING!\n");
            continue;
        }
        /// ==== all vectors legal====
        oursims[n] = sim(w1ix, w2ix);
        fprintf(stderr, "\tw2grass(%f)\n", oursims[n]);
        n++;

    }

    if (argc == 4) {
        f = fopen(argv[3], "w");
        assert(f != 0);
        for(int i = 0; i < n; ++i) {
            fprintf(f, "%f %f\n", simlexes[i], oursims[i]);
        }
        fclose(f);
    }

    return 0;
}