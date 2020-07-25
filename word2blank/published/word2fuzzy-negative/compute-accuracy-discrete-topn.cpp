// compute-accuracy  using cross entropy, but made faster by caching logs.
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <vector>
#include <algorithm>

using namespace std;

#define min(i, j) ((i) < (j) ? (i) : (j))
#define max(i, j) ((i) > (j) ? (i) : (j))

const long long max_size = 2000;         // max length of strings
const long long N = 5;                   // number of closest words
const long long max_w = 50;              // max length of vocabulary entries

void analogy(double *a, double *b, double *x, double *y, int size) {
    for(int i = 0; i < size; ++i) {
        double delta = (b[i] + x[i]) - min(b[i] + x[i], a[i]);
        y[i] = max(delta, 0.0);
        y[i] = min(delta, 1.0);
        assert(y[i] >= 0);
    }

    /*
    double total = 0;
    for(int i = 0; i < size; ++i) {
        total +=  y[i];
    }

    for(int i = 0; i < size; ++i) {
        y[i] /= total;
    }
    */
}

double entropylog(double x) {
    if (x < 1e-400L) {
        return 0;
    }
    return log(x);
}

double entropy(double *v, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i) 
        H += -v[i] * entropylog(v[i]) - (1 - v[i]) * entropylog(1 - v[i]);
    return H;
}

double crossentropy(double *v, double *lv, double *loneminusv, double *w, double *lw, double *loneminusw, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        if (w[i] < 1e-300L) w[i] = 1e-300L;
        H += v[i] * (lv[i] - lw[i]) + // (entropylog(v[i]) - entropylog(w[i])) + 
            (1 - v[i]) * (loneminusv[i] - loneminusw[i]); // (1 - v[i]) * (entropylog((1 - v[i])) - entropylog((1-w[i])));
    }
    return H;
}

double kl(double *v, double *lv, double *loneminusv, double *w, double *lw, double *loneminusw, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        if (w[i] < 1e-300L) w[i] = 1e-300L;
        // H += -v[i] * entropylog(w[i]) - (1 - v[i]) *  entropylog((1-w[i]));
        H += -v[i] * lw[i] - (1 - v[i]) *  loneminusw[i]; //entropylog((1-w[i]));
    }
    return H;
}

int main(int argc, char **argv)
{
  FILE *f;
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[N][max_size], file_name[max_size];
  double dist, bestd[N], vec[max_size], vecl[max_size], vecloneminus[max_size];
  long long words, size, a, b, c, d, b1, b2, b3, threshold = 0;
  double *M, *Ml, *Mloneminus, *Mprob;
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
  M = (double *)malloc(words * size * sizeof(double));
  Mprob = (double *)malloc(words * size * sizeof(double));
  Ml = (double *)malloc(words * size * sizeof(double));
  Mloneminus = (double *)malloc(words * size * sizeof(double));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(double) / 1048576);
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

    for (a = 0; a < size; a++) {
        float fl;
        fread(&fl, sizeof(float), 1, f);
        M[a + b * size] = fl;
        // convert these to our version.
        Mprob[a + b * size] = fl;
        Mprob[a + b * size] = pow(2.0, M[a + b * size]);
    }

    // normalize M
    /*
    float len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
    */

  }



  // now, how to discretize?
  for(a = 0; a < size; ++a) {
      vector<double> pvals;
      double total = 0;
      for(b = 0; b < words; ++b) {
          pvals.push_back(Mprob[b * size + a]);
          total += M[b * size + a];
      }

      // std::sort(pvals.begin(), pvals.end());
      // const double mid = pvals[int(pvals.size() * 0.8)];

      for(b = 0; b < words; ++b) {
          // discretize M based on Mprob
           Mprob[b * size + a] = Mprob[b * size + a] >= 1.0 ? 2 : 1;
           M[b * size + a] = M[b * size + a] >= 0 ? 1 : 0;
      }
  }

  // compress and make doubly stochastic
  for (int i = 0; i < 10; ++i) {
      for(a = 0; a < size; ++a) {
          double total = 0;
          for(b = 0; b < words; ++b) {
              total += Mprob[b * size + a];
          }

          for(b = 0; b < words; ++b) {
              Mprob[b * size + a] /= total;
              Mprob[b * size + a] = max(min(1.0, Mprob[b * size + a]), 0.0);
          }
      }

      /*
      for(b = 0; b < words; ++b) {
          double total = 0;
          for(b = 0; b < words; ++b) {
              total += Mprob[b * size + a];
          }

          for(a = 0; a < size; ++a) {
              Mprob[b * size + a] /= total;
              Mprob[b * size + a] = max(min(1.0, Mprob[b * size + a]), 0.0);
          }
      }
      */
  }

  
  // setup caches
  for(b = 0; b < words; ++b) {
      for(a = 0; a < size; ++a) {
          Ml[b * size + a] = entropylog(Mprob[b * size + a]);
          Mloneminus[b * size + a] = entropylog(1.0 - Mprob[b * size + a]);
      }
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
        fflush(stdout);
        printf("ACCURACY TOP1: %.2f %%  (%d / %d)\n", CCN / (double)TCN * 100, CCN, TCN);
        fflush(stdout);
        printf("Total accuracy: %.2f %%   Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %% \n", CACN / (double)TACN * 100, SEAC / (float)SECN * 100, SYAC / (float)SYCN * 100);
        fflush(stdout);
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

    fprintf(stderr, "%s : %s :: %s : %s?\n", st1, st2, st3, st4);
    fflush(stderr);

    // for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestd[a] = 1000;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    TQ++;
    if (b1 == words) continue;
    if (b2 == words) continue;
    if (b3 == words) continue;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st4)) break;
    if (b == words) continue;



    // for (a = 0; a < size; a++) vec[a] = (M[a + b2 * size] - M[a + b1 * size]) + M[a + b3 * size];
    analogy(&Mprob[b1 * size], &Mprob[b2 * size], &Mprob[b3 * size], vec, size);
    for(int i = 0; i < size; ++i) {
        // assert(vec[i] >= 0);
        // assert(vec[i] <= 1.1);
        vecl[i] = entropylog(vec[i]);
        vecloneminus[i] = entropylog(1 - vec[i]);
    }

    TQS++;
    for (c = 0; c < words; c++) {
      if (c == b1) continue;
      if (c == b2) continue;
      if (c == b3) continue;

      // dist = 0;
      // for (a = 0; a < size; a++) { dist += vec[a] * M[a + c * size]; }
      dist = kl(vec, vecl, vecloneminus, &Mprob[c * size], 
              &Ml[c * size], &Mloneminus[c * size], size);
      
      for (a = 0; a < N; a++) {
        // if (dist > bestd[a]) {
        if (dist < bestd[a]) {
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

    for (int i = 0; i < N; i++) {
        fprintf(stderr, "\t%20s:%f\n", bestw[i], bestd[i]);
        fflush(stderr);
    }


    for (int i = 0; i < N; ++i) {
        if (!strcmp(st4, bestw[i])) {
          fprintf(stderr, "found!\n"); fflush(stderr);
          CCN++;
          CACN++;
          if (QID <= 5) SEAC++; else SYAC++;
          break;
        }
    }

    /*
    if (!strcmp(st4, bestw[0])) {
      CCN++;
      CACN++;
      if (QID <= 5) SEAC++; else SYAC++;
    }
    */

    if (QID <= 5) SECN++; else SYCN++;
    TCN++;
    TACN++;
  }
  printf("Questions seen / total: %d %d   %.2f %% \n", TQS, TQ, TQS/(double)TQ*100);
  fflush(stdout);
  return 0;
}
