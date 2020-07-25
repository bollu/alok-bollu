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
const long long max_n = 100; // max top-n vectors asked for.

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

double crossentropyfuzzy(double *v, double *lv, double *loneminusv, double *w, double *lw, double *loneminusw, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        H += v[i] * (lv[i] - lw[i]) + // (entropylog(v[i]) - entropylog(w[i])) + 
            (1 - v[i]) * (loneminusv[i] - loneminusw[i]); // (1 - v[i]) * (entropylog((1 - v[i])) - entropylog((1-w[i])));
    }
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
        H += -v[i] * entropylog(w[i]) - (1 - v[i]) *  entropylog((1-w[i]));
    }
    return H;
}

double kl(double *v, double *lv, double *loneminusv, double *w, double *lw, double *loneminusw, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        H += v[i] * (lv[i] - lw[i]);
    }
    assert(H >= 0);
    return H;
}


const int ARGVECFILE = 1;
const int ARGQUESTIONSFILE = 2;
const int ARGN = 3;

void addphrases(const char *str, char *vocab, 
        double *M, double *Ml, double *Mloneminus,
        const int size, long long int *words) {
    // check that word has an _ in it.
    int isphrase = 0;
    for(int i = 0; i< strlen(str); ++i) {
        isphrase = isphrase || str[i] == '_';
    }

    if (!isphrase) return;

    // word was in vocab
    for(int i = 0; i < *words; ++i) {
        if (!strcmp(vocab + i * max_w, str)) return;
    }

    // OK, phrase is actually new. add to dict.
    strcpy(vocab + *words *max_w, str);

    for(int i = 0; i < size; ++i) {
        M[*words * size  + i] = 1.0;
    }

    char subphrase[max_w];
    int i = 0, bufi = 0;
    while(1) {
        bufi = 0;
        while(str[i] != '_' && str[i] != '\0') {
            subphrase[bufi] = str[i];
            bufi++; i++;
        }
        subphrase[bufi] = 0;
        fprintf(stderr, "## word: %s | subphrase: %s\n", str, subphrase);

        // now lookup word in subphrase in our corpus...
        int found = 0;
        for(int w = 0; w < *words; ++w) {
            if (strcmp(vocab + w * max_w, subphrase) == 0) { continue; }

            found = 1;
            for(int i = 0; i < size; ++i) {
                M[*words * size + i] *= M[w * size + i];
            }
            break;
        }
        if (!found) {
            // phrase does not exist in our corpus.
            fprintf(stderr, "subword out of corpus: |%s|\n", subphrase);
            assert(0 && "subphrase out of corpus!");
            // early exit, return.
            return;
        }

        if (str[i] == '\0') { break;  }
        else { assert(str[i] == '_'); i++; }
    }


    double total = 0;
    for(int i = 0; i < size; ++i) { total += M[*words * size + i]; }
    for(int i = 0; i < size; ++i) { 
        M[*words * size + i] /= total;
        Ml[*words * size + i] = log(M[*words * size + i]);
        Mloneminus[*words * size + i] = log1p(-M[*words * size + i]);
    }

    *words = *words + 1;
    return;
}


int main(int argc, char **argv)
{
  FILE *f;
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[max_n][max_size], file_name[max_size];
  double dist, bestd[max_n], vec[max_size], vecl[max_size], vecloneminus[max_size];
  long long words, size, a, b, c, d, b1, b2, b3;
  double *M, *Ml, *Mloneminus;
  char *vocab;
  int TCN, CCN = 0, TACN = 0, CACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;
  if (argc < 4) {
    printf("Usage: ./compute-accuracy <VECFILE> <QUESTIONSFILE> <N> \n"
            "- VECFILE contains word projections\n"
            "- QUESTIONSFILE contains questions.\n"
            "- N is topN closest words to try and match\n"
          );
    return 0;
  }
  strcpy(file_name, argv[ARGVECFILE]);
  long long N = atoi(argv[ARGN]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);

  const long  MAXWORDS = 2 * words;
  vocab = (char *)malloc(words * 2 * max_w * sizeof(char));
  M = (double *)malloc(words * 2 * size * sizeof(double));
  Ml = (double *)malloc(words * 2 * size * sizeof(double));
  Mloneminus = (double *)malloc(words * 2 * size * sizeof(double));

  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(double) / 1048576);
    return -1;
  }
  b = 0;
  while(!feof(f)) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
    b++;



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
  fclose(f);


  // take exponent
  for(b = 0; b < words; ++b) {
      for (a = 0; a < size; a++) { M[a + b * size] = pow(M_E, M[a + b * size]); }
  }


  /* try to build all pairs vectors. not sustainable.
  for(long long i = 0; i < words; ++i) {
      for(long long j = 0; j < words; ++j) {
          const long long ix = words + i * words + j;
          sprintf(vocab + ix*max_w, "%s_%s", vocab + i*max_w, vocab + j * max_w);
          printf("vocab[%lld] := %s\n", ix, vocab + ix*max_w);

          for(a = 0;  a < size; ++a) {
              // just multiply for intersection.
              M[ix + a] = M[i*size+a] * M[j * size+a];
          }
      }
  }
  */

  // normalize each feature across all words.
  // for(a = 0; a < size; ++a) {
  //     double total = 0;
  //     for(b = 0; b < PHRASES; ++b) {
  //         total += M[b * size + a];
  //     }

  //     for(b = 0; b < PHRASES; ++b) {
  //         M[b * size + a] /= total;
  //         M[b * size + a] = max(min(1.0, M[b * size + a]), 0.0);
  //     }
  // }

  // normalize the features across each words
  for(b = 0; b < words; ++b) {
      double total = 0;
      for(a = 0; a < size; ++a) { total += M[b * size + a]; }
      for(a = 0; a < size; ++a) { M[b * size + a] /= total; }
  }


  // now parse the input file and check if we can find the concepts.
  f = fopen(argv[ARGQUESTIONSFILE], "rb");
  if (!f) { printf("unable to open questions file: |%s|\n", argv[ARGQUESTIONSFILE]); }
  assert(f && "unable to open questions file");
  while(!feof(f)) {
      fscanf(f, "%s", st1);
      for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
      if (!strcmp(st1, ":") || !strcmp(st1, "EXIT") || feof(f)) { continue; }
      fscanf(f, "%s", st2);
      for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
      fscanf(f, "%s", st3);
      for (a = 0; a < strlen(st3); a++) st3[a] = toupper(st3[a]);
      fscanf(f, "%s", st4);
      for (a = 0; a < strlen(st4); a++) st4[a] = toupper(st4[a]);

      // split at '_' and take intersections.
      fprintf(stderr, "%s:%s :: %s:%s\n", st1, st2, st3, st4);
      assert(words < MAXWORDS);
      addphrases(st1, vocab, M, Ml, Mloneminus, size, &words);
      assert(words < MAXWORDS);
      addphrases(st2, vocab, M, Ml, Mloneminus, size, &words);
      assert(words < MAXWORDS);
      addphrases(st3, vocab, M, Ml, Mloneminus, size, &words);
      assert(words < MAXWORDS);
      addphrases(st4, vocab, M, Ml, Mloneminus, size, &words);
      assert(words < MAXWORDS);

  }
  fclose(f);
   

  for(b = 0; b < words; ++b) {
      for(a = 0; a < size; ++a) {
          Ml[b * size + a] = log(M[b * size + a]);
          // Mloneminus[b * size + a] = entropylog(1.0 - M[b * size + a]);
          Mloneminus[b * size + a] = log1p(- M[b * size + a]);
      }
  }


  printf("TopN: %d\n", (int)N);
  TCN = 0;
  f = fopen(argv[ARGQUESTIONSFILE], "rb");
  if (!f) { printf("unable to open questions file: |%s|\n", argv[ARGQUESTIONSFILE]); }
  assert(f && "unable to open questions file");
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    fscanf(f, "%s", st1);
    for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
    if ((!strcmp(st1, ":")) || (!strcmp(st1, "EXIT")) || feof(f)) {
      if (TCN == 0) TCN = 1;
      if (QID != 0) {
        fflush(stdout);
        printf("ACCURACY TOP1: %.2f %%  (%d / %d)\n", CCN / (double)TCN * 100, CCN, TCN);
        fflush(stdout);
        printf("Total accuracy: %.2f %%   Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %% \n", CACN / (double)TACN * 100, SEAC / (float)SECN * 100, SYAC / (float)SYCN * 100);
        fflush(stdout);
      }
      QID++;
      fscanf(f, "%s", st1);
      if (feof(f)) break;
      printf("%s:\n", st1);
      TCN = 0;
      CCN = 0;
      continue;
    }
    if (!strcmp(st1, "EXIT")) break;
    fscanf(f, "%s", st2);
    for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
    fscanf(f, "%s", st3);
    for (a = 0; a<strlen(st3); a++) st3[a] = toupper(st3[a]);
    fscanf(f, "%s", st4);

    for (a = 0; a < strlen(st4); a++) st4[a] = toupper(st4[a]);
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;
    b1 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st2)) break;
    b2 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st3)) break;
    b3 = b;

    fprintf(stderr, "%s : %s :: %s : %s?\n", st1, st2, st3, st4);
    fflush(stderr);

    for (a = 0; a < N; a++) bestd[a] = 1000;
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
        vecl[i] = log(vec[i]);
        vecloneminus[i] = log1p(-vec[i]);
    }

    TQS++;
    for (c = 0; c < words; c++) {
      if (c == b1) continue;
      if (c == b2) continue;
      if (c == b3) continue;
        
          ///dist = kl(vec, vecl, vecloneminus, &M[c * size], &Ml[c * size], &Mloneminus[c * size], size) +
          ///    kl(&M[c * size], &Ml[c * size], &Mloneminus[c * size], vec, vecl, vecloneminus, size);
      dist = crossentropy(vec, vecl, vecloneminus, 
              &M[c * size], &Ml[c * size], &Mloneminus[c * size],
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
