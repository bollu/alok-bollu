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

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define max_size 2000
#define N 40
#define max_w 50

FILE *f;
char st1[max_size];
char *bestw[N];
char file_name[max_size], st[100][max_size];
// float dist, len, bestd[N];
float complex dist, bestd[N];
float len;
float complex vec[max_size];
long long words, size, a, b, c, d, cn, bi[100];
float complex *M;
char *vocab;

// find dot product of two words
void dot() {
    float lensq;
    if (cn != 3) {
        printf("ERROR: expected two vectors to find dot product\n");
    }

    float d = 0;
    for (a = 0; a < size; a++)
        d += M[a + bi[1] * size] * conj(M[a + bi[2] * size]);

    lensq = 0;
    for (a = 0; a < size; a++)
        lensq += M[a + bi[2] * size] * conj(M[a + bi[2] * size]);
    d /= sqrt(lensq);

    lensq = 0;
    for (a = 0; a < size; a++)
        lensq += M[a + bi[1] * size] * conj(M[a + bi[1] * size]);
    d /= sqrt(lensq);
    lensq = 0;

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
        for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    // normalize
    float lensq = 0;
    for (a = 0; a < size; a++) lensq += vec[a] * conj(vec[a]);
    len = sqrt(lensq);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
        a = 0;
        for (b = 0; b < cn; b++)
            if (bi[b] == c) a = 1;
        if (a == 1) continue;
        dist = 0;
        for (a = 0; a < size; a++) {
            complex float new = vec[a] * conj(M[a + c * size]);
            dist += new;
        }
        for (a = 0; a < N; a++) {
            if (cabs(dist) > cabs(bestd[a])) {
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
    for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], cabs(bestd[a]));
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
    M = (float complex *)malloc((long long)words * (long long)size *
                                sizeof(float complex));
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
            float r, i;
            fread(&r, sizeof(float), 1, f);
            fread(&i, sizeof(float), 1, f);
            M[a + b * size] = r + I * i;
            // printf("%s[%lld] =  %f+%fi\n", &vocab[b * max_w], a, r, i);
        }
        //  getchar();
        float complex lensq = 0;
        // normalize
        for (a = 0; a < size; a++)
            lensq += M[a + b * size] * conj(M[a + b * size]);
        len = sqrt(lensq);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
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
