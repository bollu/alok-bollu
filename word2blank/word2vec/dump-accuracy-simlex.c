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

#define max_size 2000
#define N 40
#define max_w 50

typedef float real;

FILE *f;
char st1[max_size];
char *bestw[N];
char file_name[max_size], st[100][max_size];
float dist, len, bestd[N], vec[max_size];
long long words, size, a, b, c, d, cn, bi[100];
float *M;
char *vocab;


int main(int argc, char **argv) {
    if (argc < 3) {
        printf(
            "Usage: ./distance <VECFILE> <SIMLEXFILE\nwhere VECFILE contains word projections in "
            "the BINARY FORMAT\n");
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
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (float *)malloc((long long)words * (long long)size * sizeof(float));
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
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    fclose(f);

    f = fopen(argv[2], "r");

    // throw away first line.
    {
        char *throwaway; size_t throwi;
        getline(&throwaway, &throwi, f);
    }

    static const int MAX_LINES_SIMLEX = 1002;
    float *simlex = (float *)malloc(sizeof(float) * MAX_LINES_SIMLEX);
    float *oursim = (float *)malloc(sizeof(float) * MAX_LINES_SIMLEX);

    char word1[max_size], word2[max_size], word3[max_size];
    while(!feof(f)) {
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

        const double simlexacc = atof(word3);


        // skip word and grab simlex score;
        printf("> |%s| :: |%s| : simlex(%f)  <\n", word1, word2, simlexacc);
        free(linebuf);

        int w1ix = -1, w2ix = -1;
        for(int i = 0; i < words; ++i) {
            if (!strcmp(&vocab[max_w*i], word1)) {
                w1ix = i;
                printf("\tvocab[%d] = %s\n", w1ix, &vocab[max_w*w1ix]);
            }
            if (!strcmp(&vocab[max_w*i], word2)) {
                w2ix = i;
                printf("\tvocab[%d] = %s\n", w2ix, &vocab[max_w*w2ix]);
            }
        }

        if (w1ix == -1 || w2ix == -1) {
            printf("\tSKIPPING!\n");
        }
    }

    return 0;
}
