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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))
const long long max_size = 2000;         // max length of strings
const long long N = 1;                   // number of closest words
const long long max_w = 50;              // max length of vocabulary entries

// orthonormalize a2 wrt a1
void orthonorm(const int size, const float *a1, const float *a2, float *a2_perp) {
    // we assume a1 is orthonormal.
    float dot = 0;
    for(int i = 0; i < size; ++i) {
        dot += a1[i] * a2[i];
    }

    // compute a2' = a2 - (a2.a1) a1_hat
    // we assume that a1 is already normalized, hence a1_hat = a1
    float l = 0;
    for(int i = 0; i < size; ++i) {
        a2_perp[i] = a2[i] -  dot * a1[i];
        l += a2_perp[i] * a2_perp[i];
    }

    l = sqrt(l);
    for(int i = 0; i < size; ++i) { a2_perp[i] /= l; }


}

void principal_angles(const int size,
        const float *a1, const float *a2,
        const float *b1, const float *b2,
        float *sigma_1, float *sigma_2) {
    float a2_perp[max_size];
    float b2_perp[max_size];

    orthonorm(size, a1, a2, a2_perp);
    orthonorm(size, b1, b2, b2_perp);
    float mat[2][2];
    for(int i = 0; i < 2; ++i) {
        const float *a = i == 0 ? a1 : a2_perp;
        for(int j = 0; j < 2; ++j) {
            const float *b = j == 0 ? b1 : b2_perp;
            mat[i][j] = 0;
            for(int k = 0;  k < size; ++k) { mat[i][j] += a[k] * b[k]; }
        } // end j
    } // end i

    // SVD of a 2x2 matrix
    const float a = mat[0][0], b = mat[0][1], c = mat[1][0], d = mat[1][1];
    const float s1 = a*a + b*b + c*c + d*d;
    const float s2 = 
        sqrt((a*a + b*b - c*c - d*d) * (a*a + b*b - c*c - d*d) + 
                (4 * (a*c + b*d) * (a*c + b*d))); 
    *sigma_1 = sqrt(0.5*(s1 + s2)); 
    *sigma_2 = (s1 <= s2 ? 0 : sqrt(0.5*(s1 - s2))); 
}

int main(int argc, char **argv)
{
  FILE *f;
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[N][max_size], file_name[max_size];
  float len, bestd[N];
  long long words, size, a, b, c, d, b1, b2, b3, threshold = 0;
  float *M;
  char *vocab;
  int TCN, CCN = 0, TACN = 0, CACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;
  if (argc < 2) {
    printf("Usage: ./compute-accuracy <FILE> <threshold>\nwhere FILE contains word projections, and threshold is used to reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30000)\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  if (argc > 2) threshold = atoi(argv[2]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  if (threshold) if (words > threshold) words = threshold;
  fscanf(f, "%lld", &size);
  assert(size % 2 == 0);
  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (float *)malloc(words * size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
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
    for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    printf("lensq: %4.2f | normalized.\n", len);
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);
  
  TCN = 0;
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    scanf("%s", st1);
    for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
    if ((!strcmp(st1, ":")) || (!strcmp(st1, "EXIT")) || feof(stdin)) {
      if (TCN == 0) TCN = 1;
      if (QID != 0) {
        printf("ACCURACY TOP1: %.2f %%  (%d / %d)\n", CCN / (float)TCN * 100, CCN, TCN);
        printf("Total accuracy: %.2f %%   Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %% \n", CACN / (float)TACN * 100, SEAC / (float)SECN * 100, SYAC / (float)SYCN * 100);
      }
      QID++;
      scanf("%s", st1);
      if (feof(stdin)) break;
      printf("%s:\n", st1);
      TCN = 0;
      CCN = 0;
      continue;
    }
    if (!strcmp(st1, "EXIT")) break;
    scanf("%s", st2);
    for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
    scanf("%s", st3);
    for (a = 0; a<strlen(st3); a++) st3[a] = toupper(st3[a]);
    scanf("%s", st4);
    for (a = 0; a < strlen(st4); a++) st4[a] = toupper(st4[a]);
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;
    b1 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st2)) break;
    b2 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st3)) break;
    b3 = b;
    for (a = 0; a < N; a++) bestd[a] = -1000;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    TQ++;
    if (b1 == words) continue;
    if (b2 == words) continue;
    if (b3 == words) continue;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st4)) break;
    if (b == words) continue;
    // for (a = 0; a < size; a++) vec[a] = (M[a + b2 * size] - M[a + b1 * size]) + M[a + b3 * size];
    TQS++;
    for (c = 0; c < words; c++) {
      if (c == b1) continue;
      if (c == b2) continue;
      if (c == b3) continue; 
      
      float cw_sigma_1 = 0, cw_sigma_2 = 0;
      
      principal_angles(size,
            M + size * b1,
            M + size * c,
            M + size * b2,
            M + size * b3,
            &cw_sigma_1,
            &cw_sigma_2);
      
      const float gain = sqrt((cw_sigma_1*cw_sigma_1) + (cw_sigma_2*cw_sigma_2));
      for (int a = 0; a < N; a++) {
        if (gain > bestd[a]) {
        // if (loss < bestd[a]) {
            for (d = N - 1; d > a; d--) {
                bestd[d] = bestd[d - 1];
                strcpy(bestw[d], bestw[d - 1]);
            }
            // update best loss.
            // bestd[a] = gain;
            bestd[a] = gain;
            strcpy(bestw[a], vocab + c * max_w);
            break;
        }
      }
    }
    if (!strcmp(st4, bestw[0])) {
      CCN++;
      CACN++;
      if (QID <= 5) SEAC++; else SYAC++;
    }
    const bool correct = !strcmp(st4, bestw[0]);
    fprintf(stderr, "%15s : %15s :: %15s : %15s (correct: %15s) %5s\n", st1, st2, st3, bestw[0], st4, correct ? "✓": "x");
    if (QID <= 5) SECN++; else SYCN++;
    TCN++;
    TACN++;
  }
  printf("Questions seen / total: %d %d   %.2f %% \n", TQS, TQ, TQS/(float)TQ*100);
  return 0;
}
