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

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

float and(float r, float s) {
    return r *s;
}
float mk01(float r) {
    return powf(2, r);
}

// or: r + s - rs
// and: 1 - [(1 - r) + (1 - s) - (1 - r)(1 - s)]
// = 1 - [2 - r - s - (1 - r - s + rs)] 
// = 1 - [2 - r - s - 1 + r + s - rs] // release ()
// = 1 - [1 - rs] // simplify inside []
// = 1 - 1 + rs // release []
// = rs
#define NUM_SPARKS 100


void plotHistogram(const char *name, float *vals, int n, int nbuckets) {
    // number of values in each bucket.
    int buckets[nbuckets];
    for(int i = 0; i < nbuckets; ++i) buckets[i] = 0;

    float vmax = vals[0];
    float vmin = vals[0];
    for(int i = 0; i < n; ++i) vmax = vals[i] > vmax ? vals[i] : vmax;
    for(int i = 0; i < n; ++i) vmin = vals[i] < vmin ? vals[i] : vmin;

    float multiple = (vmax - vmin) / nbuckets;

    for(int i = 0; i < n; ++i) {
        int b = floor((vals[i] - vmin) / multiple);
        b = b >= nbuckets ? (nbuckets -1): (b < 0 ? 0 : b);
        buckets[b]++;
    }
    
    int total = 0;
    for(int i = 0; i < nbuckets; ++i) total += buckets[i];

    printf("%s: |", name);
    for(int i = 0; i < nbuckets; ++i) {
        printf(" %f ", ((buckets[i] / (float)total)) * 100.0);
    }
    printf("|");

}


int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char bestw[N][max_size];
  char file_name[max_size], st[100][max_size];
  float len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, cn, bi[100];
  float *M;
  char *vocab;
  if (argc < 2) {
    printf("Usage: ./word-analogy <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
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
    // printf("%s(NODISCRETE): ", vocab + b *max_w);
    for (a = 0; a < size; a++) {
        fread(&M[a + b * size], sizeof(float), 1, f);
        M[a + b*size] =mk01(M[a + b*size]);
        if (a < 10) printf("%4.2f ", M[a+b*size]);
    }
    printf("\n");
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size]; // * M[a + b * size];
    // len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
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
    // if (cn < 3) {
    //   printf("Only %lld words were entered.. three words are needed at the input to perform the calculation\n", cn);
    //   continue;
    // }
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = 0;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == 0) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == 0) continue;
    printf("\n                                              Word              Distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < size; a++) vec[a] = M[a + bi[0] * size];

    for (int i = 1; i < cn; ++i) {
        for (a = 0; a < size; a++) vec[a] = vec[a] * M[a + bi[i] * size];
    }

    float vecSize = 0;
    for (a = 0; a < size; a++) vecSize += vec[a]; //  * vec[a];
    // len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= vecSize;
    float vals[words];
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
      if (c == bi[0]) continue;
      if (c == bi[1]) continue;
      if (c == bi[2]) continue;
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      float intersectSize = 0;
      for (a = 0; a < size; a++) intersectSize += vec[a] * M[a + c * size];
      const float dist  = intersectSize / vecSize;
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
    printf("\n");
  }
  return 0;
}
