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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float real;  // Precision of float numbers

const long long max_size = 2000;  // max length of strings
const long long N = 40;           // number of closest words that will be shown
const long long max_w = 50;       // max length of vocabulary entries

int main(int argc, char **argv) {
    FILE *f;
    char st1[max_size];
    char *bestw[N];
    char file_name[max_size], st[100][max_size];
    float dist, metricdist, lensq, len, bestd[N], bestmetricd[N],
        vecnorm[max_size], vec[max_size];
    long long words, size, a, b, c, d, cn, bi[100];
    float *M, *Mnorm;
    real *metric;
    char *vocab;
    if (argc < 3) {
        printf(
            "Usage: ./distance  <FILE> <USEMETRIC=0,1>\nwhere FILE contains "
            "word projections in "
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

    int usemetric = atoi(argv[2]);
    assert(usemetric == 0 || usemetric == 1);
    metric = (real *)malloc(size * sizeof(real));

    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (float *)malloc((long long)words * (long long)size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }

    Mnorm = (float *)malloc((long long)words * (long long)size * sizeof(float));
    if (Mnorm == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }

    if (usemetric) {
        printf("metric:");
        for (a = 0; a < size; a++) {
            if (a % 10 == 0) printf("\n  ");
            fscanf(f, "%f", &metric[a]);
            printf("%7.2lf ", metric[a]);
        }
        printf("|\n");
        getchar();
    } else
        for (a = 0; a < size; a++) metric[a] = 1;

    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        vocab[b * max_w + a] = 0;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        lensq = 0;
        for (a = 0; a < size; a++) {
            // lensq += M[a + b * size] * metric[a] * M[a + b * size];
            lensq += M[a + b * size] * M[a + b * size];
        }
        if (lensq >= 0) {
            len = sqrt(lensq);
            // we are normalizing by length^2
            for (a = 0; a < size; a++)
                Mnorm[a + b * size] = M[a + b * size] / len;
        } else {
            for (a = 0; a < size; a++) Mnorm[a + b * size] = 0;
        }

        float pos = 0, neg = 0;
        for (a = 0; a < size; a++) {
            if (metric[a] >= 0)
                pos += M[a + b * size];
            else
                neg += M[a + b * size];
        }
        printf("%30s  M+: %5.2f  M-: %5.2f\n", &vocab[b * max_w], pos, neg);
    }
    fclose(f);
    while (1) {
        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestmetricd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        printf("Enter word or sentence (EXIT to break): ");
        // a = len(st1)
        a = 0;
        while (1) {
            st1[a] = fgetc(stdin);
            if ((st1[a] == '\n') || (a >= max_size - 1)) {
                st1[a] = 0;
                break;
            }
            a++;
        }
        if (!strcmp(st1, "EXIT")) break;
        // cn: number of words being searched
        cn = 0;
        b = 0;
        c = 0;
        while (1) {
            st[cn][b] = st1[c];
            b++;
            c++;
            st[cn][b] = 0;
            if (st1[c] == 0) break;
            if (st1[c] == ' ') {
                cn++;
                b = 0;
                c++;
            }
        }
        cn++;
        for (a = 0; a < cn; a++) {
            for (b = 0; b < words; b++)
                if (!strcmp(&vocab[b * max_w], st[a])) break;
            if (b == words) b = -1;
            bi[a] = b;
            printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
            if (b == -1) {
                printf("Out of dictionary word!\n");
                break;
            }
        }
        // bi[x]: position of st[x] in vocab
        if (b == -1) continue;
        printf(
            "\n                          Word\t\tCosine "
            "distance\t\tMetric distance"
            "\n--------------------------------------------------------"
            "----------------\n");
        for (a = 0; a < size; a++) vecnorm[a] = 0;
        for (a = 0; a < size; a++) vec[a] = 0;
        // create a vecnormtor that is the sum of all input vecnormtors
        // vecnorm[a] is the vecnormtor to find cosine sim with
        for (b = 0; b < cn; b++) {
            if (bi[b] == -1) continue;
            for (a = 0; a < size; a++) vecnorm[a] += Mnorm[a + bi[b] * size];
            for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
        }
        len = 0;
        for (a = 0; a < size; a++) {
            // len += vecnorm[a] * metric[a] * vecnorm[a];
            len += vecnorm[a] * vecnorm[a];
        }
        len = sqrt(len);
        for (a = 0; a < size; a++) vecnorm[a] /= len;
        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestmetricd[a] = 0;
        for (a = 0; a < N; a++) strcpy(bestw[a], "$$$");
        for (c = 0; c < words; c++) {
            a = 0;
            for (b = 0; b < cn; b++)
                if (bi[b] == c) a = 1;
            // if the word is in the list of words, don't take cosine sim.
            // because it will have very high cosine sim.
            if (a == 1) continue;

            dist = 0;
            metricdist = 0;
            for (a = 0; a < size; a++) {
                dist += vecnorm[a] * Mnorm[a + c * size];
            }
            for (a = 0; a < size; a += 1) {
                metricdist += vec[a] * M[a + c * size] * metric[a];
            }
            // if (metricdist < 0) continue;

            lensq = 0;
            for (int i = 0; i < size; i++) lensq += vec[a] * metric[a] * vec[a];
            // if (lensq < 0) continue;

            lensq = 0;
            for (int i = 0; i < size; i++)
                lensq += M[a + c * size] * metric[a] * M[a + c * size];
            // if (lensq < 0) continue;

            for (a = 0; a < N; a++) {
                float curd = dist;
                // if (fabs(curd) > fabs(bestd[a])) {
                if (fabs(curd) > fabs(bestd[a])) {
                    for (d = N - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        bestmetricd[d] = bestmetricd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = curd;
                    bestmetricd[a] = metricdist;
                    strcpy(bestw[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        for (a = 0; a < N; a++)
            printf("%24s\t\t%f\t\t%f\n", bestw[a], bestd[a], bestmetricd[a]);
    }
    return 0;
}
