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
// #define EXPENSIVE_CHECKS
#define max(a, b) ((a) > (b) ? (a) : (b))
typedef double real;

#define max_size 3000L         // max length of vector
#define N 1L                   // number of closest words
#define max_w 50L              // max length of vocabulary entries


// given angles, precompute sin(theta_i), cos(theta_i) and 
//  sin(theta_i) * sin(theta_{i+1}) *  ... * sin(theta_j) 0 <= i, j <= n-1
void angleprecompute(const int n, const real theta[n-1], real coss[n-1], 
        real sins[n-1], real sinaccum[n-1][n-1]) {
    for(int i = 0; i < n - 1; i++) {
        coss[i] = cos(theta[i]);
        sins[i] = sin(theta[i]);
        // cos^2 x + sin^2 x = 1
        int safe =  fabs(1.0 - (coss[i] * coss[i] + sins[i] * sins[i])) < 1e-6;
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
        float out[n]) {

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
    if(fabs(lensq - 1) >= 1e-4) { 
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
    assert(fabs(lensq - 1) < 1e-4);
    #endif
}

real lensq(const int n, const float v[n]) {
    real tot = 0;
    for (int i = 0; i < n; ++i ) tot += v[i] * v[i];
    return tot;
}

void vec2angle(const int n, const float v[n], real angles[n-1]) {
    // printf("lensq: %4.2f\n", lensq(n, v));
    // assert(fabs(1.0 - lensq(n, v)) < 1e-2);

    // convert vector to angle
    real sinprod = 1;
    for(int i = 0; i < n-1; ++i) {
        if (fabs(sinprod) < 1e-4) {
            angles[i] = 0;
        } else {
            real angle_cos = v[i] / sinprod;
            if (angle_cos < -1) angle_cos = -1;
            else if (angle_cos > 1) angle_cos = 1;
            angles[i] = acos(angle_cos);
            sinprod *= sin(angles[i]);
        }
        // printf("i: %4d v[i]: %5.2f | v[i+1]: %5.2f | angles[i]: %5.2f | sinprod: %5.2f\n", i, v[i], v[i+1],
        //         angles[i], sinprod);
    }

    angles[n-2] = atan2(v[n-1], v[n-2]);
    // printf("angles[n-2]: %5.2f\n", angles[n-2]);
    
    #ifdef EXPENSIVE_CHECKS
    float vcheck[n];
    real coss[n-1], sins[n-1], sinaccum[n-1][n-1];
    angleprecompute(n, angles, coss, sins, sinaccum);
    angle2vec(n, coss, sins, sinaccum, vcheck);
    for(int i = 0; i < n; ++i) {
        if (fabs(vcheck[i] - v[i]) > 1e-4) {
            printf("error: n: %d | i: %d | ours: %3.5f | truth: %3.5f\n" , n, i, vcheck[i], v[i]);
            assert(0);
        }
    }
    #endif

}

void analogyVec(const int n, const float v1[n], const float v2[n], const float v3[n],
        float vout[n]) {
    real a1[n-1], a2[n-1], a3[n-1], aout[n-1];
    vec2angle(n, v1, a1);
    vec2angle(n, v2, a2);
    vec2angle(n, v3, a3);

    for(int i = 0; i < n - 1; ++i) aout[i] = a2[i] - a1[i] + a3[i];
    real coss[n-1], sins[n-1], sinaccum[n-1][n-1];
    angleprecompute(n, aout, coss, sins, sinaccum);
    angle2vec(n, coss, sins, sinaccum, vout);
}



char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[N][max_size], file_name[max_size];
float bestd[N]; float vec[max_size];
real vec_angle[max_size];
real vec_coss[max_size-1], vec_sins[max_size-1], vec_sinaccum[max_size-1][max_size-1];

int main(int argc, char **argv)
{
    // fprintf(stderr, "not normalizing\n");
  FILE *f;
  long long words, size, a, b, c, d, b1, b2, b3, threshold = 0;
  float *M;
  real *Mangle, *Mlen, dist, len; 
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
  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (float *)malloc(words * size * sizeof(float));
  Mangle = (real *)malloc(words * (size - 1) * sizeof(real));
  Mlen = (real *)malloc(words * sizeof(real));
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
    Mlen[a] = len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;

    fprintf(stderr, "converting: |%20s| len: |%4.2f| \n", &vocab[b*max_w], len);
    if (len == 0) {
        for(int i = 0; i < size -1; ++i) { Mangle[b*(size-1) +i] = 0; }
    } else { 
        vec2angle(size, &M[b*size], &Mangle[b*(size-1)]);
    }
  }
  fclose(f);
  TCN = 0;
  int progress = 1;
  while (1) {
      progress++;
      if (progress % 1000 == 0) {
          progress = 1;
          // fprintf(stderr, "."); fflush(stderr);
      }

    for (a = 0; a < N; a++) bestd[a] = -9999;
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
    for (a = 0; a < N; a++) bestd[a] = -9999;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    TQ++;
    if (b1 == words) continue;
    if (b2 == words) continue;
    if (b3 == words) continue;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st4)) break;
    if (b == words) continue;
    // for (a = 0; a < size; a++) vec[a] = (M[a + b2 * size] - M[a + b1 * size]) + M[a + b3 * size];
    // analogyVec(size, &M[b1 * size], &M[b2 * size], &M[b3 * size], vec);
    for (a = 0; a < size-1; a++) {
        vec_angle[a] = (Mangle[a + b2 * size] - Mangle[a + b1 * size]) + Mangle[a + b3 * size];
        // vec_angle[a] = M_PI - fabs(fabs(vec_angle[a]) - M_PI);
    }
    angleprecompute(size, vec_angle, vec_coss, vec_sins, vec_sinaccum);
    angle2vec(size, vec_coss, vec_sins, vec_sinaccum, vec);

    TQS++;
    for (c = 0; c < words; c++) {
      if (c == b1) continue;
      if (c == b2) continue;
      if (c == b3) continue;
      // vector based
      const int VECTOR_BASED_DIFF = 0;
      if (VECTOR_BASED_DIFF) {
          dist = 0;
          for(a = 0; a < size; ++a) { dist += vec[a] * M[a+c*size]; }
      } else {
          dist = 0;
          for(a = 0; a < size-1; ++a) {
              real angle = (Mangle[a+b2*(size-1)] - Mangle[a+b1*(size-1)]) - (Mangle[a+c*(size-1)] - Mangle[a+b3*(size-1)]);
              // real angle = baseangle[a] - Mangle[a + c*size];
              angle = M_PI - fabs(fabs(angle) - M_PI);
              // we want "large" distances to win. large distances are better?
              dist += cos(angle);
          }
      }

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
    if (!strcmp(st4, bestw[0])) {
      CCN++;
      CACN++;
      if (QID <= 5) SEAC++; else SYAC++;
    }
    if (QID <= 5) SECN++; else SYCN++;
    TCN++;
    TACN++;
  }
  printf("Questions seen / total: %d %d   %.2f %% \n", TQS, TQ, TQS/(float)TQ*100);
  return 0;
}
