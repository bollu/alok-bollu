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

const long long max_size = 2000;  // max length of strings
const long long N = 40;           // number of closest words that will be shown
const long long max_w = 50;       // max length of vocabulary entries

int main(int argc, char **argv) {
    FILE *f;
    char st1[max_size];
    char bestw[N][max_size];
    char file_name[max_size], st[100][max_size];
    float dist, len, bestd[N];
    Vec v;
    Vec *M;
    long long words, size, a, b, c, d, cn, bi[100];
    char *vocab;
    if (argc < 2) {
        printf(
            "Usage: ./word-analogy <FILE>\nwhere FILE contains word "
            "projections in the BINARY FORMAT\n");
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
    M = (Vec *)malloc((long long)words *
                      sizeof(Vec));  // (long long)size * sizeof(float));
    v.alloc(size);

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
        len = 0; 
        for (a = 0; a < size/2; a++) len += M[b].v[a] * M[b].v[a];
        len = sqrt(len);
        for (a = 0; a < size/2; a++) M[b].v[a] /= len;
    }


    fclose(f);
    while (1) {
        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        printf("Enter three words (EXIT to break): ");
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
        if (cn < 3) {
            printf(
                "Only %lld words were entered.. three words are needed at the "
                "input to perform the calculation\n",
                cn);
            continue;
        }
        for (a = 0; a < cn; a++) {
            for (b = 0; b < words; b++)
                if (!strcmp(&vocab[b * max_w], st[a])) break;
            if (b == words) b = 0;
            bi[a] = b;
            printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
            if (b == 0) {
                printf("Out of dictionary word!\n");
                break;
            }
        }
        if (b == 0) continue;
        printf(
            "\n                                              Word              "
            "Distance\n--------------------------------------------------------"
            "----------------\n");

        for(int i = 0; i < size; ++i) {
            v.v[i] = M[bi[1]].v[i] - M[bi[0]].v[i] + M[bi[2]].v[i];
        }

        len = 0;
        for(int i = 0; i < size/2; ++i) {
            len += v.v[i] * v.v[i];
        }

        for(int i = 0; i < size/2; ++i) {
            v.v[i] /= len;
        }


        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        for (c = 0; c < words; c++) {
            if (c == bi[0]) continue;
            if (c == bi[1]) continue;
            if (c == bi[2]) continue;
            a = 0;
            for (b = 0; b < cn; b++)
                if (bi[b] == c) a = 1;
            if (a == 1) continue;

            dist = 0;
            for(int i = 0; i < size/2; ++i) {
                dist += v.v[i] * M[c].v[i];
            }

            dist += dotSymplectic(size/2, v.v + size/2, M[c].v + size/2);

            // dist = 0;
            // for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
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
    }
    return 0;
}
