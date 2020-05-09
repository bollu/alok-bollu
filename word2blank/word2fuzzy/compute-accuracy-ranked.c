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

#define min(i, j) ((i) < (j) ? (i) : (j))
#define max(i, j) ((i) > (j) ? (i) : (j))

const long long max_size = 2000;         // max length of strings
const long long max_w = 50;              // max length of vocabulary entries
const long long max_n = 800; // max top-n vectors asked for.

// A    :   B  :: X    : Y?
// man : king :: woman: queen?
// king - man + woman
// B/A U X
// (B U X) \cap A^C) 
void analogy(double *a, double *b, double *x, double *y, int size) {
    for(int i = 0; i < size; ++i) {
        // ---- 2. (better default)----
        double delta = (b[i] + x[i]) - min(b[i] + x[i], a[i]);
        
        // ---- 3. very very close but slightly worse :(
        // double b_plus_x = b[i] + x[i];
        // double delta = b_plus_x - min(b_plus_x, a[i]);
        
        // ------4.
        //double b_minus_a = b[i]  - min(b[i], a[i]);
        //double x_minus_a = x[i]  - min(x[i], a[i]);
        //double delta = b_minus_a + x_minus_a - b_minus_a * x_minus_a;
        y[i] = delta;
        assert(y[i] >= 0);
        assert(y[i] <= 1);
    }
    // double total = 0;
    // for(int i = 0; i < size; ++i) { total +=  y[i]; }
    // for(int i = 0; i < size; ++i) { y[i] /= total; }
}


double entropy(double *v, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i) 
        H += -v[i] * log(v[i]) - (1 - v[i]) * log(1 - v[i]);
    return H;
}

double crossentropyfuzzy(double *v, double *lv, double *loneminusv, double *w, double *lw, double *loneminusw, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        H += v[i] * (lv[i] - lw[i]) + // (entropylog(v[i]) - entropylog(w[i])) + 
            (1 - v[i]) * (loneminusv[i] - loneminusw[i]); // (1 - v[i]) * (entropylog((1 - v[i])) - entropylog((1-w[i])));
    }

    if (H < 0) {
        fprintf(stderr, "H: %4.2f\n", H); fflush(stderr);
    } 
    assert(H == H);
    assert(H >= 0);
    return H;
}

double crossentropy(double *v, double *lv, double *loneminusv, double *w, double *lw, double *loneminusw, int size) {
    double H = 0;
    // pi log qi
    for(int i = 0; i < size; ++i)  {
        H -= v[i] * lw[i];
    }
    assert(H >= 0);
    return H;
}

double klfuzzy(double *v, double *lv, double *loneminusv, double *w, double *lw, double *loneminusw, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        // H += -v[i] * entropylog(w[i]) - (1 - v[i]) *  entropylog((1-w[i]));
        H += -v[i] * lw[i] - (1 - v[i]) *  loneminusw[i];
    }
    assert(H >= 0);
    return H;
}

double kl(double *v, double *lv, double *loneminusv, double *w, double *lw, double *loneminusw, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        // H += -v[i] * entropylog(w[i]) - (1 - v[i]) *  entropylog((1-w[i]));
        H += v[i] * (lv[i] - lw[i]);
    }
    if (H < 0) {
        fprintf(stderr, "H: %4.2f\n", H); fflush(stderr);
    }
    assert(H >= 0);
    return H;
}

int main(int argc, char **argv)
{
  FILE *f;
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[max_n][max_size], file_name[max_size];
  double dist, bestd[max_n], vec[max_size], vecl[max_size], vecloneminus[max_size];
  long long words, size, a, b, c, d, b1, b2, b3;
  double *M, *Ml, *Mloneminus;
  char *vocab;
  int TCN, TACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;
  double CCN = 0, CACN = 0;
  if (argc < 2) {
    printf("Usage: ./compute-accuracy <FILE> <N>\n"
            "- FILE contains word projections\n"
            );
    return 0;
  }
  strcpy(file_name, argv[1]);
  const long long N = 200; assert(N <= max_n);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (double *)malloc(words * size * sizeof(double));
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
    }

    double len = 0;
    for (a = 0; a < size; a++) { len += M[a + b * size] * M[a + b * size]; }
    len = sqrt(len);
    for (a = 0; a < size; a++) { M[a + b * size] /= len; }

  }


  // take exponent
  for(b = 0; b < words; ++b) {
      for (a = 0; a < size; a++) { M[a + b * size] = pow(M_E, M[a + b * size]); }
  }

  // normalize the features of each word
  //#for(b = 0; b < words; ++b) {
  //#    double total = 0;
  //#   for(a = 0; a < size; ++a) { total += M[b * size + a]; }
  //#    for(a = 0; a < size; ++a) {
  //#        M[b * size + a] /= total;
  //#    }
  //#}


  // normalize each feature of all words
  for(a = 0; a < size; ++a) {
      double total = 0;
      for(b = 0; b < words; ++b) { total += M[b * size + a]; }
      for(b = 0; b < words; ++b) { M[b * size + a] /= total; }
  }



  for(b = 0; b < words; ++b) {
      for(a = 0; a < size; ++a) {
          Ml[b * size + a] = log(M[b * size + a]);
          // Mloneminus[b * size + a] = entropylog(1.0 - M[b * size + a]);
          Mloneminus[b * size + a] = log1p(- M[b * size + a]);
      }
  }

  fclose(f);
  printf("TopN: %d\n", N);
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
        printf("ACCURACY TOP1: %.2f %%  (%4.2f / %d)\n", CCN / (double)TCN * 100, CCN, TCN);
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

    for (a = 0; a < N; a++) bestd[a] = 99999;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    TQ++;
    if (b1 == words) continue;
    if (b2 == words) continue;
    if (b3 == words) continue;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st4)) break;
    if (b == words) continue;



    // for (a = 0; a < size; a++) vec[a] = (M[a + b2 * size] - M[a + b1 * size]) + M[a + b3 * size];
    analogy(&M[b1 * size], &M[b2 * size], &M[b3 * size], vec, size);
    for(int i = 0; i < size; ++i) {
        // assert(vec[i] >= 0);
        // assert(vec[i] <= 1.1);
        // vecl[i] = entropylog(vec[i]);
        // vecloneminus[i] = entropylog(1 - vec[i]);
        assert(vec[i] >= 0);
        assert(vec[i] <= 1);
        vecl[i] = vec[i] == 0 ? 0 : log(vec[i]);
        vecloneminus[i] = (1 - vec[i]) == 0 ? 0 : log1p(-vec[i]);
    }

    TQS++;
    for (c = 0; c < words; c++) {
      if (c == b1) continue;
      if (c == b2) continue;
      if (c == b3) continue;
      dist = crossentropyfuzzy(vec, vecl, vecloneminus, 
              &M[c * size], &Ml[c * size], &Mloneminus[c * size],
              size) + 
          crossentropyfuzzy(&M[c * size], &Ml[c * size], &Mloneminus[c * size],
                  vec, vecl, vecloneminus, 
                  size);


      for (a = 0; a < N; a++) {
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
          fprintf(stderr, "\tfound!\n"); fflush(stderr);
          CCN += 1.0 / (i + 1);
          CACN += 1.0 / (i + 1);
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
