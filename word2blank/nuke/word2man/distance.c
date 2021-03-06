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

FILE *f;
char st1[max_size];
char *bestw[N];
char file_name[max_size], st[100][max_size];
float dist, len, bestd[N], vec[max_size];
long long words, size, a, b, c, d, cn, bi[100];
float *M, *sqrtm;
float *TransformedM;
char *vocab;

// find dot product of two words
void dot() {
    float lensq;
    if (cn != 3) {
        printf("ERROR: expected two vectors to find dot product\n");
    }

    float d = 0;
    for (a = 0; a < size; a++) d += (a == 0 ? -1 : 1) * M[a + bi[1] * size] * M[a + bi[2] * size];

    // lensq = 0;
    // for (a = 0; a < size; a++)
    //     lensq += M[a + bi[2] * size] * M[a + bi[2] * size];
    // d /= sqrt(lensq);

    // lensq = 0;
    // for (a = 0; a < size; a++)
    //     lensq += M[a + bi[1] * size] * M[a + bi[1] * size];
    // d /= sqrt(lensq);
    // lensq = 0;

    printf("dot: %f\n", d);
}

void cosine() {
    printf(
        "\n                                              Word       "
        "Cosine "
        "distance\n----------------------------------------------------"
        "----"
        "----------------\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
        if (bi[b] == -1) continue;
        for (a = 0; a < size; a++) vec[a] += TransformedM[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len +=  (c == 0 ? -1 : 1) * vec[a] * vec[a];
    assert(len >= 0);
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;

    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
        dist = 0;
        for (a = 0; a < size; a++) dist += vec[a] * TransformedM[a + c * size];
        for (a = 0; a < N; a++) {
            if ((dist) > (bestd[a])) {
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
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (float *)malloc((long long)words * (long long)size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }

    TransformedM = NULL;
    TransformedM = (float *)malloc((long long)words * (long long)size * sizeof(float));
    if (TransformedM == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }

    sqrtm = (float *)malloc((long long)size * (long long)size * sizeof(float));
    if (sqrtm == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)size * size * sizeof(float) / 1048576, size, size);
        return -1;
    }

    char token[100];
    fscanf(f, "%s", token);
    printf("token: (%s)\n", token);
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            // fread(&sqrtm[i*size+j], sizeof(float), 1, f);
            fscanf(f, "%f", &sqrtm[i*size+j]);
            printf("%f ", sqrtm[i*size+j]);
        }
        printf("\n\t");
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

        for(int i = 0; i < size; ++i) {
            TransformedM[b * size +i] = 0;
            for(int j = 0; j < size; ++j) {
                TransformedM[b * size +i] += sqrtm[i * size + j] * M[b * size + j];
            }
        }

        len = 0;
        for (a = 0; a < size; a++) len += TransformedM[b * size + a] * TransformedM[b * size + a];
        // printf("%s\n", &vocab[b * max_w]);
        // printf("len: %3.2f\n", len);
        if (len < 0) len = 0;
        assert(len >= 0);
        len = sqrt(len);
        for (a = 0; a < size; a++) TransformedM[a + b * size] /= len;
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
