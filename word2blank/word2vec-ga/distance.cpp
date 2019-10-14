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
#include "vec.h"

#define max_size 2000
#define N 40
#define max_w 50
#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))

FILE *f;
char st1[max_size];
char *bestw[N];
char file_name[max_size], st[100][max_size];
float dist, len, bestd[N];
Vec vec;
long long words, size, a, b, c, d, cn, bi[100];
Vec *M;
char *vocab;
real *quadform;
real *Ay;

void plotHistogram(const char *name, real *vals, int n, int nbuckets) {
    // number of values in each bucket.
    int buckets[nbuckets];
    for(int i = 0; i < nbuckets; ++i) buckets[i] = 0;

    real vmax = vals[0];
    real vmin = vals[0];
    for(int i = 0; i < n; ++i) vmax = vals[i] > vmax ? vals[i] : vmax;
    for(int i = 0; i < n; ++i) vmin = vals[i] < vmin ? vals[i] : vmin;

    real multiple = (vmax - vmin) / nbuckets;

    for(int i = 0; i < n; ++i) {
        int b = floor((vals[i] - vmin) / multiple);
        b = b >= nbuckets ? (nbuckets -1): (b < 0 ? 0 : b);
        buckets[b]++;
    }
    
    int total = 0;
    for(int i = 0; i < nbuckets; ++i) total += buckets[i];

    printf("%s: |", name);
    for(int i = 0; i < nbuckets; ++i) {
        printf(" %f ", ((buckets[i] / (real)total)) * 100.0);
    }
    printf("|\n");

}

// find dot product of two words
void dot() {
    float lensq;
    if (cn != 3) {
        printf("ERROR: expected two vectors to find dot product\n");
    }

    float d = 0;
    // for (a = 0; a < size; a++) d += M[a + bi[1] * size] * M[a + bi[2] *
    // size];
    d = mulQuadForm(size, M[bi[1]].v, quadform, M[bi[2]].v,  Ay, nullptr);

    float l1 = mulQuadForm(size, M[bi[1]].v, quadform, M[bi[1]].v,  Ay, nullptr);
    float l2 = mulQuadForm(size, M[bi[2]].v, quadform, M[bi[2]].v,  Ay, nullptr);

    d /= sqrt(fabs(l1));
    d /= sqrt(fabs(l2));

    printf("dot: %f\n", d);
}

real getNormalizationFactorL(Vec w) {
    // this . dot(other)
    float maxdot = 0;
    for(int i = 0; i < N; ++i) {

        const float  dist = mulQuadForm(size, w.v, quadform, M[i].v,  Ay, nullptr);
        maxdot = max(maxdot, fabs(dist));
    }
    return maxdot;
}

real getNormalizationFactorR(Vec w) {
    // others . dot (this)
    float maxdot = 0;
    for(int i = 0; i < N; ++i) {
        const float  dist = mulQuadForm(size, M[i].v, quadform, w.v,  Ay, nullptr);
        maxdot = max(maxdot, fabs(dist));
    }
    return maxdot;
}

void cosine() {
    real vals[words];

    
    {
        printf(
                "\n                                              Word       "
                "Cosine "
                "distance\n----------------------------------------------------"
                "----"
                "----------------\n");
        // for (a = 0; a < size; a++) vec[a] = 0;
        vec.fillzero();
        vec.accumscaleadd(1.0, M[bi[0]]);

        // for (b = 0; b < cn; b++) {
        //     if (bi[b] == -1) continue;
        //     vec.accumscaleadd(1.0, M[bi[b]]);
        //     // for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
        // }
        for (int i = 0; i < min(size, 10); i++) {
            printf("%3.2f  ", vec.ix(i));
        }
        printf("\n");
        len = 0;


        // for (a = 0; a < size; a++) len += vec[a] * vec[a];
        // len = sqrt(len);
        // for (a = 0; a < size; a++) vec[a] /= len;
        // vec.normalize();
        for (a = 0; a < N; a++) bestd[a] = -1;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        for (c = 0; c < words; c++) {
            a = 0;
            for (b = 0; b < cn; b++)
                if (bi[b] == c) a = 1;
            if (a == 1) continue;
            // dist = 0;
            // for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
            // dist = vec.dotContainmentConstrained(M[c],  0, 2, 0, 2, nullptr, nullptr);
            // const float curnorm = getNormalizationFactorR(c);
            dist = mulQuadForm(size, vec.v, quadform, M[c].v, Ay, nullptr);
            float l1 = getNormalizationFactorL(vec);
            float l2 = getNormalizationFactorR(M[c]);

            dist /= sqrt(fabs(l1));
            dist /= sqrt(fabs(l2));
            vals[c] = dist;

            for (a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (d = N - 1; d > a; d--) {
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
        plotHistogram("distances", vals, words, 10);
    }


    {
        printf(
                "\n                                              Word       "
                "Cosine "
                "distance\n----------------------------------------------------"
                "----"
                "----------------\n");
        // for (a = 0; a < size; a++) vec[a] = 0;

        vec.fillzero();
        vec.accumscaleadd(1.0, M[bi[0]]);

        len = 0;
        // for (a = 0; a < size; a++) len += vec[a] * vec[a];
        // len = sqrt(len);
        // for (a = 0; a < size; a++) vec[a] /= len;
        // vec.normalize();
        for (a = 0; a < N; a++) bestd[a] = -1;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        for (c = 0; c < words; c++) {
            a = 0;
            for (b = 0; b < cn; b++)
                if (bi[b] == c) a = 1;
            if (a == 1) continue;
            // dist = 0;
            // for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
            dist = mulQuadForm(size, M[c].v, quadform, vec.v, Ay, nullptr);
            float l1 = getNormalizationFactorL(M[c]);
            float l2 = getNormalizationFactorR(vec);

            dist /= sqrt(fabs(l1));
            dist /= sqrt(fabs(l2));
            vals[c] = dist;

            for (a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (d = N - 1; d > a; d--) {
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

        plotHistogram("distances", vals, words, 10);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf(
            "Usage: ./distance <FILE>\nwhere FILE contains word projections in "
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

    printf("setting up quadform...\n");
    quadform = (float*)malloc(sizeof(float) * size * size);
    setupDotContainmentMat(size, quadform);
    printf("setting up Ay...\n");
    Ay = (float *)malloc(sizeof(float) * size * size);


    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (Vec *)malloc((long long)words * sizeof(Vec));
    vec.alloc(size);
    // (long long)size * sizeof(float));
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
        M[b].alloc(size);
        readvec(f, M[b]);
        // M[b].normalize();
        printf("%s:", vocab + b * max_w);
        // printf(" lensq: %f  ", M[b].lensq(quadform));
        for (int i = 0; i < min(10, size); i++) {
            printf("%f ", M[b].ix(i));
        }
        printf("\n");
        // for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1,
        // f); len = 0; for (a = 0; a < size; a++) len += M[a + b * size] * M[a
        // + b * size]; len = sqrt(len); for (a = 0; a < size; a++) M[a + b *
        // size] /= len;
    }

    fclose(f);
    while (1) {
        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        printf("Enter word or sentence (EXIT to break): ");
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
            if (b == -1 && strcmp(st[0], "DOT") != 0) {
                printf("Out of dictionary word!\n");
                break;
            }
        }
        if (b == -1 && strcmp(st[0], "DOT") != 0) continue;

        if (!strcmp(st[0], "DOT")) {
            dot();
        } else {
            cosine();
        }
    }
    return 0;
}
