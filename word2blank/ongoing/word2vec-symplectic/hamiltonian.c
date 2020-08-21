// Attempt to recreate 

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

static const long long max_size = 2000;         // max length of strings
const long long N = 10;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries


int main(int argc, char **argv) {
    srand(0);
    FILE *f;
    char st1[max_size];
    char *bestw[N];
    char file_name[max_size], st[100][max_size];
    float dist, mom, len, disp, delmom, deldist, bestd[N], vec[max_size];
    long long words, size, a, b, c, d, cn, bi[100];
    float *M;
    float *lens;
    char *vocab;
    if (argc < 2) {
        printf("Usage: ./hamiltonian <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
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
    // Read Vectors 
    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        vocab[b * max_w + a] = 0;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
      
        // Normalize `p`
        len = 0;
        for (a = 0; a < size/2; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size/2; a++) M[a + b * size] /= len;
        
        // Normalize `q`
        len = 0;
        for (a = size/2; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = size/2; a < size; a++) M[a + b * size] /= len;
        
        for(a = 0; a < size; a++) {
            if (M[a + b * size] != M[a + b * size]) { 
                printf("nan |%20s|[%4lld]\n", vocab + b * max_w, a);
            }
        }
    }
    fclose(f);

    long long NWORDS = 5; // number of words being studied for this purpose
    double dt = 1e-3; // value of dt
    long long NITERS = 10; // number of iterations
    long long posns[NWORDS];
    for (long long j = 0; j < NWORDS; j++)  posns[j] = 1 + rand() % (words-1);    // random position for a given word

    for (long long i = 0; i < NITERS; i++)  {
      for (b = 0; b < words; b++) {
        for (a = 0; a < size/2; a++) M[(a + size/2) + b * size] += M[a + b * size] * dt;  // dq += p * dt
      }
      // finding similar words
      for (long long j = 0; j < NWORDS; j++)  { 
        printf("\nWord: %s\n", vocab + posns[j] * max_w);

        for (a = 0; a < size; a++) vec[a] = 0;  // cleaning the vec array
        for (a = 0; a < size; a++) vec[a] += M[a + posns[j] * size];
        dist = 0, mom = 0;
        for (a = 0; a < size/2; a++) dist += vec[a] * vec[a];
        for (a = size/2 ; a < size; a++) mom += vec[a] * vec[a];
        dist = sqrt(dist);
        mom = sqrt(mom);
        for (a = 0; a < size/2; a++) vec[a] /= dist;
        for (a = size/2; a < size; a++) vec[a] /= mom;
        
        for (a = 0; a < N; a++) bestd[a] = -1;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        for (c = 0; c < words; c++) {
          a = 0;
          if (posns[j] == c) a = 1;
          if (a == 1) continue;
          deldist = 0, delmom = 0;
          for (a = 0; a < size/2; a++) deldist += vec[a] * M[a + c * size];
          for (a = size/2; a < size; a++) delmom += vec[a] * M[a + c * size];
          // disp = func(deldist, delmom);
          disp = deldist + delmom;
          for (a = 0; a < N; a++) {
            if (disp > bestd[a]) {
              for (d = N - 1; d > a; d--) {
                bestd[d] = bestd[d - 1];
                strcpy(bestw[d], bestw[d - 1]);
              }
              bestd[a] = disp;
              strcpy(bestw[a], &vocab[c * max_w]);
              break;
            }
          }
        }
        for (a = 0; a < N; a++) printf("%30s\n", bestw[a]);
      }
      printf("\n\n------------------------------------\n\n");
    }
    return 0;
}
