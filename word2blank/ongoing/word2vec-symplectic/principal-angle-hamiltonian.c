// Attempt to recreate `hamiltonian.py` in C

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
const long long N = 10;                         // number of closest words that will be shown
const long long max_w = 50;                     // max length of vocabulary entries
static const long long NWORDS = 5;             // number of words being studied for this purpose
static const double dt = 10;                  // value of dt
static const long long NITERS = 5;             // number of iterations

// float func(float a, float b);                   // f(distance, momentum)

int main(int argc, char **argv) {
    FILE *f;
    char *bestw[N];
    char file_name[max_size];
    float dist, mom, len, disp, delmom, deldist, bestd[N];
    long long words, size, a, b, c, d;
    float *M;
    char *vocab;
    if (argc < 3) {
        printf("Usage: ./hamiltonian <FILE> <SEED>\nwhere FILE contains word projections in the BINARY FORMAT\nand SEED is a random seed\n");
        return 0;
    }
    srand(atoi(argv[2]));         // random seed for reproducibility
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

    long long posns[NWORDS];
    for (long long j = 0; j < NWORDS; j++)  posns[j] = 1 + rand() % (words-1);    // random position for a given word

    for (long long num = 0; num < NITERS; num++)  {
      for (b = 0; b < words; b++) {
        for (a = 0; a < size/2; a++) M[(a + size/2) + b * size] += M[a + b * size] * dt;  // dq += p * dt
      }
      // finding similar words
      for (long long uix = 0; uix < NWORDS; uix++)  { 
        printf("\nWord: %s\n", vocab + posns[uix] * max_w);

        float bestgains[N];
        char bestws[N][max_w];
        for(int i = 0; i < N; ++i)  { bestgains[i] = -1; bestws[i][0] = '\0'; }

        for(int wix = 0; wix < words; ++wix) {
          if uix
          float *uvecs[2] = { M + posns[uix] * size, M + posns[uix] * size + (size/2)};
          float *wvecs[2] = { M + wix * size, M + wix * size + (size/2)};

          float mat[2][2];
          for(int i = 0; i < 2; ++i) {
            for(int j = 0; j < 2; ++j) {
              mat[i][j] = 0;
              for(int k = 0;  k < size/2; ++k) {
                mat[i][j] += uvecs[i][k] * wvecs[j][k];
              } // end k
            } // end j
          } // end i


          // singular values: https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
          const float a = mat[0][0], b = mat[0][1], c = mat[1][0], d = mat[1][1];
          const float s1 = a*a + b*b + c*c + d*d;
          const float s2 = 
              sqrt((a*a + b*b - c*c - d*d) * (a*a + b*b - c*c - d*d) + 
                    (4 * (a*c + b*d) * (a*c + b*d)));

          const float sigma_1 = sqrt(0.5*(s1 + s2));
          const float sigma_2 = (s1 <= s2 ? 0 : sqrt(0.5*(s1 - s2)));
          // TODO: can improve this by not taking sqrt and then squaring.
          // -ffast-math removes this, IIUC.
          const float gain = sigma_1*sigma_1 + sigma_2*sigma_2;

          for(int i = 0; i < N; ++i) {
            // sorted in descending order. bestgains[0] > bestgains[1] > ...
            if(gain > bestgains[i]) {
              // move everything to the right.
              for(int j = N - 1; j > i; j--) {
                bestgains[j] = bestgains[j-1];
                strcpy(bestws[j], bestws[j-1]);
              }
              // update best gain.
              bestgains[i] = gain;
              strcpy(bestws[i], vocab + wix * max_w);
              break;
            } // end gain > bestgains[i]
          } // end N loop
        }
        printf("===closest to %s===\n", vocab + posns[uix]*max_w);
        for(int i = 0; i < N; ++i) {
          printf("%20s | %4.2f\n", bestws[i], bestgains[i]);
        }
      }
      printf("\n\n------------------------------------\n\n");
    }
    return 0;
}

// float func(float a, float b)
// {
//   float c = a + b;
//   return c;
// }

