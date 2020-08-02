// Show that the dot products EXACTLY CANCEL! WTF.
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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#define __USE_GNU
#include <fenv.h>


static const long long max_size = 2000;         // max length of strings
const long long N = 50;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries


int main(int argc, char **argv) {
    // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW ); 
    // fesetexceptflag(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW ); 
    FILE *f;
    char st1[max_size];
    char *bestw[N];
    char file_name[max_size], st[100][max_size];
    float dist, len, bestd[N], vec[max_size];
    long long words, size, a, b, c, d, cn, bi[100];
    float *M;
    float *lens;
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
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (float *)malloc((long long)words * (long long)size * sizeof(float));
    lens = (float *)malloc((long long)words * (long long)2 * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
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
        for (a = 0; a < size/2; a++) len += M[a + b * size] * M[a + b * size];
        lens[b*2] = len = sqrt(len);
        for (a = 0; a < size/2; a++) M[a + b * size] /= len;
        // printf("|%20s| lensq: %4.2f\n", vocab + b*max_w, len);

        len = 0;
        for (a = size/2; a < size; a++) len += M[a + b * size] * M[a + b * size];
        lens[b*2+1] = len = sqrt(len);
        for (a = size/2; a < size; a++) M[a + b * size] /= len;
        // printf("|%20s| lensq: %4.2f\n", vocab + b*max_w, len);

        // printf("checking for NaN: |%20s| \n", vocab + b*max_w);
        for(a = 0; a < size; a++) {
            if (M[a+b*size] != M[a+b*size]) { 
                printf("nan |%20s|[%4lld]\n", vocab + b*max_w, a);
                // assert(0);
            }
            // assert(fabs(M[a+b*size]) < 1.0);
        }
    }
    fclose(f);
    long long nzero = 0, total = 0;
    for (c = 1; c < words; c++) {
        for (int c2 = c+1; c2 < words; c2++) {

            float deltas[2][max_size];
            /*
            for(int i = 0; i < 2; ++i) {
                const int OFFSET = i*size/2;
                for(a = 0; a < size/2; a++) {
                    deltas[i][a] = M[OFFSET + c2*size] - M[OFFSET + a+c*size];
                    printf("deltas[%3d][%3d] = %5.4d\n", i, a, deltas[i][a]);
                }
            }
            */

            float dot[2];
            dot[0] = dot[1] = 0;
            for(a = 0; a < size/2; a++) {
                dot[0] += deltas[0][a] * deltas[1][a+size/2];
                // printf("dot[0]: %20.4d\n", dot[0]);
                dot[1] += deltas[0][a+size/2]* deltas[1][a];
            }

            const float delta = fabs(dot[0] - dot[1]);
            if (delta < 1e-3) {
                nzero++;
            } else {
                printf("====\n");
                printf("||%20s | %20s||\n-- δ: %20.4f \n-- dot0: %20.4f\n-- dot1: %20.4f \n", 
                        vocab + max_w*c, 
                        vocab + max_w*c2,
                        dot[0] - dot[1], 
                        dot[0], dot[1]);
                assert(0);
            }
            total++;
        }
    }
    return 0;
}
