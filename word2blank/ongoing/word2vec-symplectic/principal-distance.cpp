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

/*
    // this code is useful for word2vec angles
    char str_us[512][2], str_vs[512][2]
    scanf("%s %s %s %s", str_us[0[]], str_us[1], str_vs[0], str_vs[1]);
    int us[2], vs[2];
    for(int i = 0; i < words; ++i) {
      if (!strcmp(vocab + b*i, str_us[0])) { us[0] = i; }
      if (!strcmp(vocab + b*i, str_us[1])) { us[1] = i; }
      if (!strcmp(vocab + b*i, str_vs[0])) { vs[0] = i; }
      if (!strcmp(vocab + b*i, str_vs[1])) { vs[1] = i; }

      float matuv[2][2];
      for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
          for(int k = 0;  k < size; ++k) {

          }
        }
      }
    }
    */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

const long long max_size = 2000;         // max length of strings
const long long TOPK = 20;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

float func(float a, float b);            // f(distance, momentum)

int main(int argc, char **argv) {
  FILE *f;
  char file_name[max_size];
  long long words, size, a, b;
  float *M;
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
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
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

    float len = 0;
    for (a = 0; a < size/2; a++) len += M[a + b * size] * M[a + b * size];
    // printf("lengths %15s:  %4.2f | ", vocab + b*max_w, len);
    len = sqrt(len);
    
    if(len != 0) { for (a = 0; a < size/2; a++) M[a + b * size] /= len; }

    len = 0;
    for (a = size/2; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    // printf("%4.2f\n", len);
    if(len != 0) {  for (a = size/2; a < size; a++) M[a + b * size] /= len; }
  }
  fclose(f);
  while (1) {
    printf("Enter u: ");
    char str_u[max_w]; scanf("%s", str_u);
    int uix = -1;
    for(int i = 0; i < words; ++i) {
      if (!strcmp(vocab + max_w*i, str_u)) { uix = i; }
    } // end words
    if (uix == -1) { printf("unable to find word: |%s|\n", str_u); continue; }

    float bestgains[TOPK];
    char bestws[TOPK][max_w];
    for(int i = 0; i < TOPK; ++i)  { bestgains[i] = -1; bestws[i][0] = '\0'; }

    for(int wix = 0; wix < words; ++wix) {
      float *uvecs[2] = { M + uix * size, M + uix * size + (size/2)};
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

      printf("    - dots: [%4.2f | %4.2f | %4.2f | %4.2f]"
             " s1: %4.2f | s2: %4.2f\n",
             mat[0][0], mat[0][1], mat[1][0], mat[0][1],
             s1, s2);


      const float sigma_1 = sqrt(0.5*(s1 + s2));
      const float sigma_2 = (s1 <= s2 ? 0 : sqrt(0.5*(s1 - s2)));
      // TODO: can improve this by not taking sqrt and then squaring.
      // -ffast-math removes this, IIUC.
      const float gain = sigma_1*sigma_1 + sigma_2*sigma_2;

      
      for(int i = 0; i < TOPK; ++i) {
        // sorted in descending order. bestgains[0] > bestgains[1] > ...
        if(gain > bestgains[i]) {
          // move everything to the right.
          for(int j = TOPK-1; j > i; j--) {
            bestgains[j] = bestgains[j-1];
            strcpy(bestws[j], bestws[j-1]);
          }
          // update best gain.
          bestgains[i] = gain;
          strcpy(bestws[i], vocab + wix*max_w);
          break;
        } // end gain > bestgains[i]
      } // end TOPK loop
    } // end words

    printf("===closest to %s===\n", vocab + uix*max_w);
    for(int i = 0; i < TOPK; ++i) {
      printf("%20s | %4.2f\n", bestws[i], bestgains[i]);
    }

  } // end while(1)
}
