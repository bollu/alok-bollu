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

#define max(x, y) ((x) > (y) ? (x):(y))
#define min(x, y) ((x) < (y) ? (x):(y))

typedef float real;

#define EXPENSIVE_CHECKS

// given angles, precompute sin(theta_i), cos(theta_i) and 
//  sin(theta_i) * sin(theta_{i+1}) *  ... * sin(theta_j) 0 <= i, j <= n-1
void angleprecompute(const int n, const real theta[n-1], real coss[n-1], 
        real sins[n-1], real sinaccum[n-1][n-1]) {
    for(int i = 0; i < n - 1; i++) {
        coss[i] = cos(theta[i]);
        sins[i] = sin(theta[i]);
        // cos^2 x + sin^2 x = 1
        int safe =  fabs(1.0 - (coss[i] * coss[i] + sins[i] * sins[i])) < 1e-2;
        if (!safe) {
            printf("theta: %f | real:%f / coss: %f | real: %f / sins: %f\n", theta[i], 
                    cos(theta[i]), coss[i], sin(theta[i]), sins[i]);
            assert(0);
        }
    }
    
    // check interval [i..j]
    for(int i = 0; i < n - 1; ++i) {
        // j < i
        for(int j = 0; j < i; ++j) { sinaccum[i][j] = 1; }
        //j = i
        sinaccum[i][i] = sins[i];
        // j > 1
        for(int j = i + 1; j < n - 1; ++j) {
            sinaccum[i][j] = sins[j] * sinaccum[i][j-1];
        }
    }
}

// convert angles to vectors for a given index
void angle2vec(const int n, const real coss[n - 1], const real sins[n - 1], const real sinaccum[n-1][n-1],
        real out[n]) {

    // reference
    // x1          = c1
    // x2          = s1 c2
    // x3          = s1 s2 c3
    // x4          = s1 s2 s3 c4
    // x5          = s1 s2 s3 s4 c5
    // x6 = xfinal = s1 s2 s3 s4 s5
    for(int i = 0; i < n; i++) {
        out[i] = (i == 0 ? 1 : sinaccum[0][i-1]) * (i == n-1 ? 1 : coss[i]);
    }

    #ifdef EXPENSIVE_CHECKS
    real lensq = 0;
    for(int i = 0; i < n; i++) {
        lensq += out[i] * out[i];
    }
    if(fabs(lensq - 1) >= 0.2) { 
        printf("lensq: %f\n", lensq);
        printf("  cos: ["); 
        for(int i = 0; i < n; ++i) {
            printf("%f ", coss[i]);
        }
        printf("]\n"); 
        printf("  sin: ["); 
        for(int i = 0; i < n; ++i) {
            printf("%f ", sins[i]);
        }
        printf("]\n"); 
        printf("  vec: ["); 
        for(int i = 0; i < n; ++i) {
            printf("%f ", out[i]);
        }
        printf("]\n"); 
    }
    assert(fabs(lensq - 1) < 0.2);
    #endif
}

void vec2angle(int n, real vec[n], real angles[n-1]) {
    float lensq = 0;
    for(int i = 0; i < n; ++i) {
        lensq += vec[i] * vec[i];
    }

    const float len = sqrt(lensq);
    for(int i = 0; i < n; ++i) {
        vec[i] /= len;
    };

    // n = 5
    // x0 = cos t0
    // x1 = sin t0 cos t1
    // x2 = sin t0 sin t1 cos t2
    // x3 = sin t0 sin t1 sin t2 cos t3
    // x4 = sin t0 sin t1 sin t2 sin t3
    //
    // x4/x3 = sin t3 / cos t3 = tan t3
    angles[n-2] = atan2(vec[n-1], vec[n-2]);
    // to compute t2, we need to take atan2(x3, x2 * cos(t3))
    for(int i = n - 3; i >= 0; i--) {
        angles[i] =  atan2(vec[i+1], vec[i] * cos(angles[i+1]));
    }

    #ifdef EXPENSIVE_CHECKS
    real sins[n-1], coss[n-1], sinaccum[n-1][n-1];
    real vecinv[n];
    angleprecompute(n, angles, coss, sins, sinaccum);
    angle2vec(n, coss, sins, sinaccum, vecinv);

    float lensq_vecinv = 0;
    for(int i = 0; i < n; ++i) {
        lensq_vecinv += vecinv[i] * vecinv[i];
    }

    assert (fabs(lensq_vecinv - 1) < 1e-2);

    //TODO: find out why we have sign differences.
    for(int i = n - 1; i >= 0; --i) {
        if (fabs(vecinv[i] - vec[i]) > 1e-2) {
            printf("mismatch(n=%d): expected[%d] = %f | found[%d] = %f\n",  n,
                    i, vec[i], i, vecinv[i]);
            printf("ground:   [");
            for(int j = max(0, i - 10); j <= min(n-1, i + 10); ++j) {
                printf("%10.2f", vec[j]);
                           }
            printf("]\n");

            printf("computed: [");
            for(int j = max(0, i - 10); j <= min(n-1, i + 10); ++j) {
                printf("%10.2f", vecinv[j]);
                           }
            printf("]\n");
            assert(0 && "mismatch in expected and recovered vector");
        }
    }
    #endif

}

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char st1[max_size];
  char bestw[N][max_size];
  char file_name[max_size], st[100][max_size];
  float dist, len, bestd[N], vec[max_size];
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
  // a : b :: x : ?
  real sins[size-1];
  real coss[size-1];
  real sinaccum[size-1][size-1];
  real anglesA[size-1];
  real anglesB[size-1];
  real anglesX[size-1];
  real anglesY[size-1];
  real anglesZ[size-1];

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
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
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
    if (cn < 3) {
      printf("Only %lld words were entered.. three words are needed at the input to perform the calculation\n", cn);
      continue;
    }
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
    vec2angle(size, &M[a+bi[1]*size], anglesA);
    vec2angle(size, &M[a+bi[0]*size], anglesB);
    vec2angle(size, &M[a+bi[2]*size], anglesX);

    for (a = 0; a < size - 1; a++) 
        anglesY[a] = -1.0 * (anglesA[a] - anglesB[a] + anglesX[a]);


    angleprecompute(size, anglesY, coss, sins, sinaccum);
    angle2vec(size, coss, sins, sinaccum, vec);

    // compute correct vec vector
    for (a = 0; a < size; a++) vec[a] = 
      M[a + bi[1] * size] - 
      M[a + bi[0] * size] + 
      M[a + bi[2] * size];

    vec2angle(size, vec, anglesZ);

    float delta = 0;
    for(int i = 0; i < size - 1; ++i) {
        delta = fabs(anglesZ[i] - anglesY[i]);
    }
    delta /= (size - 1);
    printf("mean angle deviation between Y and Z: %.4f\n", delta);

    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
      if (c == bi[0]) continue;
      if (c == bi[1]) continue;
      if (c == bi[2]) continue;
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
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
