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

#define min(x, y) (x < y ? x : y)
#define max(x, y) (x > y ? x : y)

#define max_size  2000         // max length of strings
#define max_w  50              // max length of vocabulary entries
#define max_n  80000L              // max # of top-N vectors
typedef  long long ll;

// compound_lemma	compound_surface	compound_lemmapos	headLemma	modifierLemma	nAnnot	avgHead	stdevHead	avgModifier	stdevModifier	compositionality	stdevHeadModifier	equivalents
// ancient_history	ancient_history	ancient/a_history/n	history	ancient	18	2.7222	2.1090	1.8333	1.3827	1.7778	1.4371	old news(6);in the past(3);past(2);forever ago(2);bygone days(2);history(2);ancient knowledge(1);long time ago(1);fossil ages(1);distant(1);null(1);water under the bridge(1);over with(1);irrelevant(1);moot(1);generations ago(1);useless(1);past history(1);years ago(1);completed(1);historical relic(1);recent news(1);outdated(1);old history(1);older history(1);lost(1);not of importance(1);past times(1);extended time(1);forgotten(1);b.c(1);ancient times(1)

// 1. compound_lemma: ancient_history	
// 2. compound_surface: ancient_history	
// 3. compund_lemmapos: ancient/a_history/n	
// 4. headLemma: history	
// 5. modifierLemma: ancient	
// 6. nAnnot: 18	
// 7. avgHead: 2.7222	
// 8. stdevHead: 2.1090	
// 9. avgModified: 1.8333	
// 10, stdevModifier: 1.3827	
// 11. compositionality: 1.7778	 **
// 12. stdevHeadModifier: 1.4371	
// 13. equivalents: old news(6);in the past(3);past(2);forever ago(2);bygone days(2);history(2);ancient knowledge(1);long time ago(1);fossil ages(1);distant(1);null(1);water under the bridge(1);over with(1);irrelevant(1);moot(1);generations ago(1);useless(1);past history(1);years ago(1);completed(1);historical relic(1);recent news(1);outdated(1);old history(1);older history(1);lost(1);not of importance(1);past times(1);extended time(1);forgotten(1);b.c(1);ancient times(1)

typedef double real;

real len = 0;
char st2[max_size], st3[max_size], st4[max_size], bestw[max_n][max_size], file_name[max_size];
long long words, size, a, b, c, d, b1, b2, b3;
real *M, *Ml, *Mloneminus;
char *vocab;

const int ARGVECFILE = 1;
const int ARGCOMPOSITIONFILE = 2;
const int ARGSIMTYPE = 3;

const int LINE_HEAD_LEMMA_IX = 3;
const int LINE_MODIFIER_LEMMA_IX = 4;
const int LINE_COMPOSITIONALITY_IX = 10; 

// spearman correlation: SpearmanrResult(correlation=0.49864574933074557, pvalue=5.678722392285724e-07)
// pearson correlation: (0.4378033056059104, 1.5944978133473975e-05)

double entropyfuzzy(double *v, double *lv, double *loneminusv, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i) 
        H += -v[i] * lv[i] - (1 - v[i]) * loneminusv[i];
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

double klfuzzy(double *v, double *lv, double *loneminusv, double *w, double *lw, double *loneminusw, int size) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        // H += -v[i] * entropylog(w[i]) - (1 - v[i]) *  entropylog((1-w[i]));
        H += -v[i] * lw[i] - (1 - v[i]) *  loneminusw[i];
    }
    assert(H >= 0);
    return H;
}
void parse(char *in, char *outs[13]) {
	char buf[10000L];
	int nsections = 0;

	for(ll i = 0; i < (ll)strlen(in);) {
		ll ix = 0;
    while(1) {
			buf[ix] = toupper(in[i]);
			if(buf[ix] == '\t' || buf[ix] == '\n' || buf[ix] == '\0') { break; }
			else { ix++; i++; }
		}
		buf[ix] = '\0';
		outs[nsections] = strdup(buf);
		i++; nsections++;
	}
	assert(nsections == 13);
}



int main(int argc, char **argv) {
// /home/bollu/work/alok-bollu/word2blank/dataset-composition

  FILE *f;
  fprintf(stderr,
          "Are you sure you wish to run this? You proabbly want to run"
         "./compute-composition.py.\n");
  if (argc < 4) {
    fprintf(stderr,
            "Usage: ./compute-accuracy <VECFILE> <SIMTYPE>\n"
            "- VECFILE contains word projections\n"
            "- <COMPOSITIONFILE> is the path to the composition data\n"
            "- <SIMTYPE> is the type of similarity metric to use\n"
            "  + klheadmod for K-L(head, mod)\n"
            "  + xhheadmod for cross(x) entropy(h)(head, mod)\n"
            "  + xhsym for symmetric cross entropy\n"
          );
    return 0;
  }
  strcpy(file_name, argv[ARGVECFILE]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    fprintf(stderr, "Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words); 
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (real *)malloc(words * size * sizeof(real));
  Ml = (real *)malloc(words * size * sizeof(real));
  Mloneminus = (real *)malloc(words * size * sizeof(real));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(real) / 1048576);
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
        float fl; fread(&fl, sizeof(float), 1, f);
        M[a + b * size] = fl;
    }
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;


    // take exponent
    for (a = 0; a < size; a++) { M[a + b * size] = pow(2.0, M[a + b * size]); }
  }
  fclose(f);


  // normalize across our words
  /*
  for(b = 0; b < words; ++b) {
      double total = 0;
      for(a = 0; a < size; ++a) {
          total += M[b * size + a];
      }

      for(a = 0; a < size; ++a) {
          M[b * size + a] /= total;
          M[b * size + a] = max(min(1.0, M[b * size + a]), 0.0);
      }
  }
  */

  // normalize across our features.
  for(a = 0; a < size; ++a) {
      double total = 0;
      for(b = 0; b < words; ++b) { total += M[b * size + a]; }
      for(b = 0; b < words; ++b) { M[b * size + a] /= total; }
  }

  for(b = 0; b < words; ++b) {
      for(a = 0; a < size; ++a) {
          Ml[b * size + a] = log(M[b * size + a]);
          Mloneminus[b * size + a] = log1p(-M[b * size + a]);
      }
  }

  // 2. read composition file and evaluate composition
  f = fopen(argv[ARGCOMPOSITIONFILE], "rb");
  if (!f) { 
      fprintf(stderr, "Unable to find composition file: |%s|", 
              argv[ARGCOMPOSITIONFILE]); 
      fflush(stderr);
      return -1;
  }
  ll read = 0;
  char *line = NULL; size_t linesize = 0;
  while((read = getline(&line, &linesize, f)) != -1) {
    char *outs[13];
    parse(line, outs);
    fprintf(stderr, 
      "- head:%s modifier:%s compositionality:%5.4f\n", 
      outs[LINE_HEAD_LEMMA_IX],
      outs[LINE_MODIFIER_LEMMA_IX],
      atof(outs[LINE_COMPOSITIONALITY_IX]));

    int w1ix = -1, w2ix = -1;

    for(int i = 0; i < words; ++i) {
      if (!strcmp(outs[LINE_HEAD_LEMMA_IX], &vocab[i*max_w])) {w1ix = i;}
      if (!strcmp(outs[LINE_MODIFIER_LEMMA_IX], &vocab[i*max_w])) {w2ix = i;}
    }

    if (w1ix == -1) { fprintf(stderr, "\t-unable to find |%s|\n", outs[LINE_HEAD_LEMMA_IX]); }
    if (w2ix == -1) { fprintf(stderr, "\t-unable to find |%s|\n", outs[LINE_MODIFIER_LEMMA_IX]); }
    if (w1ix == -1 || w2ix == -1) { continue; }
    assert(w1ix >= 0); assert(w2ix >= 0);

    real *v1 = M + size * w1ix;
    real *l1 = Ml + size * w1ix;
    real *loneminus1 = Mloneminus + size *w1ix;

    real *v2 = M + size * w2ix;
    real *l2 = Ml + size * w2ix;
    real *loneminus2 = Mloneminus + size *w2ix;

    double simval = -1;
    if (!strcmp(argv[ARGSIMTYPE], "xhsym")) {
        simval = 100 -1 * (crossentropyfuzzy(v2, l2, loneminus2, v1, l1, loneminus1, size) + 
            crossentropyfuzzy(v1, l1, loneminus1, v2, l2, loneminus2, size));
    } else if (!strcmp(argv[ARGSIMTYPE], "xhheadmod")) {
        simval = -crossentropyfuzzy(v1, l1, loneminus1, v2, l2, loneminus2,  size);
    } else if (!strcmp(argv[ARGSIMTYPE], "klheadmod")) {
        simval = -crossentropyfuzzy(v1, l1, loneminus1, v2, l2, loneminus2,  size);
    } else {
        assert(false && "unknown sim type");
    }
    fprintf(stdout, 
          "%f %f\n", 
          atof(outs[LINE_COMPOSITIONALITY_IX]), 
          simval);
  }
  free(line);
  fclose(f);

  return 0;
}
