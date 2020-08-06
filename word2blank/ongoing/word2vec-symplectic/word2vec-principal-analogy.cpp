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

// singular values: https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
/*
void singular_values(float *mat22, float *sigma_1, float *sigma_2) {
    const float a = mat22[0], b = mat22[1], c = mat22[2], d = mat22[3];
    const float s1 = a*a + b*b + c*c + d*d;
    const float s2 = 
        sqrt((a*a + b*b - c*c - d*d) * (a*a + b*b - c*c - d*d) + 
                (4 * (a*c + b*d) * (a*c + b*d)));
    *sigma_1 = sqrt(0.5*(s1 + s2));
    *sigma_2 = (s1 <= s2 ? 0 : sqrt(0.5*(s1 - s2)));
}
*/

void principal_angles(int size, float *u, float *w, float *sigma_1, float *sigma_2) {
    float *uvecs[2] = { u, u + (size/2)};
    float *wvecs[2] = { w, w + (size/2)};

    float mat[2][2];
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            mat[i][j] = 0;
            for(int k = 0;  k < size/2; ++k) {
                mat[i][j] += uvecs[i][k] * wvecs[j][k];
            } // end k
        } // end j
    } // end i

    const float a = mat[0][0], b = mat[0][1], c = mat[1][0], d = mat[1][1];
    const float s1 = a*a + b*b + c*c + d*d;
    const float s2 = 
        sqrt((a*a + b*b - c*c - d*d) * (a*a + b*b - c*c - d*d) + 
                (4 * (a*c + b*d) * (a*c + b*d)));
    *sigma_1 = sqrt(0.5*(s1 + s2));
    *sigma_2 = (s1 <= s2 ? 0 : sqrt(0.5*(s1 - s2)));
}

float func(float a, float b);            // f(distance, momentum)

int main(int argc, char **argv) {
  FILE *f;
  char file_name[max_size];
  long long words, size;
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
  for (int b = 0; b < words; b++) {
    int a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);

    float len = 0;
    for (a = 0; a < size/2; a++) len += M[a + b * size] * M[a + b * size];
    printf("lengths %15s:  %4.2f | ", vocab + b*max_w, len);
    len = sqrt(len);
    
    if(len != 0) { for (a = 0; a < size/2; a++) M[a + b * size] /= len; }

    len = 0;
    for (a = size/2; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    printf("%4.2f\n", len);
    if(len != 0) {  for (a = size/2; a < size; a++) M[a + b * size] /= len; }
  }
  fclose(f);
  while (1) {
    printf("Enter [a, b, c] for  a:b::c:?: ");
    char str_a[max_w], str_b[max_w], str_c[max_w]; scanf("%s %s %s", str_a, str_b, str_c);
    int aix = -1, bix=-1, cix =-1;
    for(int i = 0; i < words; ++i) {
      if (!strcmp(vocab + max_w*i, str_a)) { aix = i; }
      if (!strcmp(vocab + max_w*i, str_b)) { bix = i; }
      if (!strcmp(vocab + max_w*i, str_c)) { cix = i; }
    } // end words
    if (aix == -1) { printf("unable to find word: |%s|\n", str_a); }
    if (bix == -1) { printf("unable to find word: |%s|\n", str_b); }
    if (cix == -1) { printf("unable to find word: |%s|\n", str_c); }

    float ab_sigma_1, ab_sigma_2;
    principal_angles(size, M + max_size*aix, M + max_size*bix,
        &ab_sigma_1, &ab_sigma_2);


    float best_losses[TOPK];
    char bestws[TOPK][max_w];
    for(int i = 0; i < TOPK; ++i)  { best_losses[i] = -1; bestws[i][0] = '\0'; }

    for(int wix = 0; wix < words; ++wix) {
    float cw_sigma_1 = 0, cw_sigma_2 = 0;
      principal_angles(size, M + max_size*cix, M + max_size*wix,
              &cw_sigma_1, &cw_sigma_2);
      // TODO: can improve this by not taking sqrt and then squaring.
      // -ffast-math removes this, IIUC.
      const float l1 = (ab_sigma_1 - cw_sigma_1);
      const float l2 = (ab_sigma_2 - cw_sigma_2);
      const float loss = l1*l1 + l2*l2;

      
      for(int i = 0; i < TOPK; ++i) {
        // sorted in ascending order: best_losses[0] < best_losses[1] < ...
        if(loss < best_losses[i]) {
          // move everything to the right.
          for(int j = TOPK-1; j > i; j--) {
            best_losses[j] = best_losses[j-1];
            strcpy(bestws[j], bestws[j-1]);
          }
          // update best gain.
          best_losses[i] = loss;
          strcpy(bestws[i], vocab + wix*max_w);
          break;
        } // end gain > best_losses[i]
      } // end TOPK loop
    } // end words

    printf("===closest to analogy===\n");
    for(int i = 0; i < TOPK; ++i) {
      printf("%20s | %4.2f\n", bestws[i], best_losses[i]);
    }

  } // end while(1)
}
