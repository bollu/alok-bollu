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
    size_t n;

    // throw away first line.
    {
        char *throwaway;
        getline(&throwaway, &n, f);
    }

    char word1[max_size], word2[max_size];
    while(!feof(f)) {
        char *linebuf = 0;
        getline(&linebuf, &n, f);
        if (strlen(linebuf) == 0) { break; }
        int i = 0;
        while(linebuf[i] != '\t' && linebuf[i] != ' ') {
            word1[i] = linebuf[i];
            i++;
        }
        const int startix = i+1;
        int j = 0;
        word1[i] = '\0';
        while(linebuf[startix + j] != '\t' && 
                linebuf[startix +j] != ' '  && linebuf[startix + j] != '\0') {
            word2[j] = linebuf[startix + j];
            j++;
        }
        word2[j] = '\0';
        printf("> |%s| :: |%s| < \n", word1, word2);
        free(linebuf);
    }

    return 0;
}
