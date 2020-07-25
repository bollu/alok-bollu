//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define min(i, j) ((i) < (j) ? (i) : (j))
#define max(i, j) ((i) > (j) ? (i) : (j))

#define max_size 2000
#define N 25
#define max_w 100

typedef double real;

FILE *f;
char st1[max_size];
char *bestw[N];
char file_name[max_size], st[100][max_size];
long long words, size, a, b, c, d, cn, bi[100];
real *M, *Ml, *Mloneminus;
char *vocab;

real entropylog(real x) {
    return log(x);
}

real entropy(real *v, int size) {
    real H = 0;
    for(int i = 0; i < size; ++i) 
        H += -v[i] * entropylog(v[i]) - (1 - v[i]) * entropylog(1 - v[i]);
    return H;
}

real fuzzycrossentropy(real *v, real *lv, real *loneminusv, real *w, real *lw, real *loneminusw, int size) {
    real H = 0;
    for(int i = 0; i < size; ++i)  {
        H += v[i] * (lv[i] - lw[i]) + (1 - v[i]) * (loneminusv[i] - loneminusw[i]); // (1 - v[i]) * (entropylog((1 - v[i])) - entropylog((1-w[i])));
    }
    return H;
}


real kl(real *v, real *lv, real *loneminusv, real *w, real *lw, real *loneminusw, int size) {
    real H = 0;
    for(int i = 0; i < size; ++i)  {
        // H += -v[i] * entropylog(w[i]) - (1 - v[i]) *  entropylog((1-w[i]));
        H += -v[i] * log(w[i]) - (1 - v[i]) *  log1p(-w[i]);
    }
    return H;
}

real sim_kl(int w1, int w2) {
    return kl(M + size * w1, Ml + size * w1, Mloneminus + size *w1, 
            M + size * w2, Ml + size * w2, Mloneminus + size *w2,
            size) + kl(M + size * w2, Ml + size * w2, Mloneminus + size *w2, 
                M + size * w1, Ml + size * w1, Mloneminus + size *w1,
                size);
}

real sim_cross_entropy(int w1, int w2) {
    return fuzzycrossentropy(M + size * w1, Ml + size * w1, Mloneminus + size *w1, 
            M + size * w2, Ml + size * w2, Mloneminus + size *w2,
            size);
}

static const int ARG_VECFILE = 1;
static const int ARG_SIMLEXFILE = 2;
static const int ARG_KL_CROSSENTROPY = 3;
static const int ARG_OUTFILE = 4;

int main(int argc, char **argv) {

    printf("are you sure you want to run this? You likely wish to run spearman.py.\n");
    if (argc < 4) {
        printf(
                "Usage: ./distance <VECFILE> <SIMLEXFILE> ['kl/'crossentropy'] [OUTFILE]"
                "\nwhere VECFILE contains word projections in the BINARY FORMAT"
                "\nSIMLEXFILE is the SimLex-999.txt from SimLex"
                "\n[OUTFILE] is the optional file to dump <simlexscore>:<vecscore>");
        return 0;
    }
    strcpy(file_name, argv[ARG_VECFILE]);
    f = fopen(file_name, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }

    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (real *)malloc((long long)words * (long long)size * sizeof(real));
    Ml = (real *)malloc(words * size * sizeof(real));
    Mloneminus = (real *)malloc(words * size * sizeof(real));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
                (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
                (long long)words * size * sizeof(float) / 1048576, words, size);
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
        for (a = 0; a < size; a++) { 
            float fl; 
            fread(&fl, sizeof(float), 1, f);
            M[a + b * size] = fl;

        }

        real len = 0;
        for (a = 0; a < size; a++) { len += M[a + b * size] * M[a + b * size]; }
        len = sqrt(len);
        for (a = 0; a < size; a++) { M[a + b * size] /= len; }

        // take exponent
        for (a = 0; a < size; a++) { M[a + b * size] = pow(2.0, M[a + b * size]); }

    }
    fclose(f);

    for(b = 0; b < words; ++b) {
        double total = 0;
        for(a = 0; a < size; ++a) {
            total += M[b * size + a];
        }

        for(a = 0; a < size; ++a) {
            M[b * size + a] /= total;
            M[b * size + a] = max(min(1.0, M[b * size + a]), 0.0);
        }
    }


    // normalize across our features.
    // for(a = 0; a < size; ++a) {
    //     double total = 0;
    //     for(b = 0; b < words; ++b) {
    //         total += M[b * size + a];
    //     }

    //     for(b = 0; b < words; ++b) {
    //         M[b * size + a] /= total;
    //         M[b * size + a] = max(min(1.0, M[b * size + a]), 0.0);
    //     }
    // }


    for(b = 0; b < words; ++b) {
        for(a = 0; a < size; ++a) {
            Ml[b * size + a] = entropylog(M[b * size + a]);
            Mloneminus[b * size + a] = entropylog(1.0 - M[b * size + a]);
        }
    }

    // open simlex file to read
    f = fopen(argv[ARG_SIMLEXFILE], "r");

    // throw away first line.
    {
        char *throwaway = NULL; size_t throwi = 0;
        getline(&throwaway, &throwi, f);
    }

    static const int MAX_LINES_SIMLEX = 1002;
    real *simlexes = (real *)malloc(sizeof(real) * MAX_LINES_SIMLEX);
    real *oursims = (real *)malloc(sizeof(real) * MAX_LINES_SIMLEX);

    char word1[max_size], word2[max_size], word3[max_size];
    int n = 0;
    for(; !feof(f);) {
        // word1\tword2\tPOS[1letter]\tSimLex999\t
        char *linebuf = NULL;
        size_t linelen = 0;
        const int read = getline(&linebuf, &linelen, f);
        if (read == -1) { break; }
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
        assert(oursims[n] >= 0);
        if (!strcmp(argv[ARG_KL_CROSSENTROPY], "kl")) {
            oursims[n] = 100 - sim_kl(w1ix, w2ix);

        } else if (!strcmp(argv[ARG_KL_CROSSENTROPY], "crossentropy")) {
            oursims[n] = 100 - sim_cross_entropy(w1ix, w2ix);

        } else { 
            fprintf(stderr, "unknown option for type of divergence: |%s|\n", argv[ARG_KL_CROSSENTROPY]);
            assert(0 && "unknown divergence type.");
        }
        n++;
    }
    if (argc == 5) {
        f = fopen(argv[4], "w");
        assert(f != 0);
        for(int i = 0; i < n; ++i) {
            fprintf(f, "%f %f\n", simlexes[i], oursims[i]);
        }
        fclose(f);
    }
    return 0;
}
