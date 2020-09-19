#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <armadillo>
#include <iostream>
#include <cassert>

#define DEBUG_LINE if(0) { printf("%s:%d\n", __FUNCTION__, __LINE__); }
//compute-accuracy /path/to/model.bin < questions-words.txt > output-file.txt
using namespace std;
using namespace arma;

const long long max_size = 2000;         // max length of strings
const long long N = 1;                  // number of closest words
const long long P = 1;
const long long max_w = 50;              // max length of vocabulary entries
arma::fcube c_syn0; 


arma::fmat log_map(const arma::fmat start, const arma::fmat end, float &L) 
{
    DEBUG_LINE
	const long long int n = start.n_rows;
    const long long int p = start.n_cols;
    DEBUG_LINE
    assert((long long int)end.n_rows == n);
    assert((long long int)end.n_cols == p);
    DEBUG_LINE
    arma::fmat I(n,n); I.eye();
    DEBUG_LINE
    arma::fmat PI_K = I - (start*arma::trans(start));
	DEBUG_LINE
    arma::fmat K = end*arma::inv(arma::trans(start)*end);
    DEBUG_LINE
    arma::fmat G = PI_K*K;
    arma::fmat U, V; arma::fvec s;
    DEBUG_LINE
	arma::svd_econ(U, s, V, G);
    DEBUG_LINE
    arma::fvec theta = arma::atan(s);
    DEBUG_LINE
    L = sqrt(arma::accu(theta % theta));
    DEBUG_LINE
    arma::fmat T_A = U * arma::diagmat(theta) * V.t();
    return T_A;  
}


arma::fmat parallel(const arma::fmat start, const arma::fmat end, const arma::fmat tgtStart, float L) 
{
    DEBUG_LINE
	const long long int n = start.n_rows;
    const long long int p = start.n_cols;
    DEBUG_LINE
    assert((long long int)end.n_rows == n);
    assert((long long int)end.n_cols == p);
    DEBUG_LINE
    arma::fmat I(n,n); I.eye();
    DEBUG_LINE
    arma::fmat PI_K = I - (start*arma::trans(start));
	DEBUG_LINE
    arma::fmat K = end*arma::inv(arma::trans(start)*end);
    DEBUG_LINE
    arma::fmat G = PI_K*K;
    arma::fmat U, V; arma::fvec s;
    DEBUG_LINE
	arma::svd_econ(U, s, V, G);
    DEBUG_LINE
    arma::fvec theta = arma::atan(s);
    DEBUG_LINE
    arma::fmat tgt_move = (-start*V*arma::diagmat(arma::sin(theta))*U.t()*tgtStart) + (U*arma::diagmat(arma::cos(theta))*U.t()*tgtStart) + ((I - (U*U.t()))*tgtStart);
    DEBUG_LINE
    //arma::fmattgt_end =  tgt_move*tgtStart;
    DEBUG_LINE
    return tgt_move;
}

arma::fmat exp_map(const arma::fmat start, const arma::fmat tgt, float L) 
{
    arma::fmat U, V; arma::fvec s;
    DEBUG_LINE
    //arma::fmatact_tgt = tgt;//*L;
	arma::svd_econ(U, s, V, tgt);
    DEBUG_LINE
    arma::fmat end = start*V*arma::diagmat(arma::cos(s))*V.t() + U*arma::diagmat(arma::sin(s))*V.t();
    DEBUG_LINE
    //end = arma::orth(end);
    DEBUG_LINE
    return end;
}

float getNaturalDist(arma::fmat &X, arma::fmat &Y) {
	const long long int n = X.n_rows;
  const long long int p = X.n_cols;
  assert((long long int)Y.n_rows == n);
  assert((long long int)Y.n_cols == p);
  arma::fvec s = arma::svd(X.t() * Y);
  s = arma::acos(s);
  return sqrt(arma::accu(s % s));
}

float getChordalDist(arma::fmat &X, arma::fmat &Y) {

	const long long int n = X.n_rows;
  const long long int p = X.n_cols;
  assert((long long int)Y.n_rows == n);
  assert((long long int)Y.n_cols == p);
  arma::fmat Proj = X*X.t() - Y*Y.t();
  return (arma::norm(Proj, "fro")/sqrt(2));
}


int main(int argc, char **argv)
{
  FILE *f;
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[N][max_size], file_name[max_size];
  float dist, bestd[N];
  long long words, size, b, c, b1, b2, b3, threshold = 0;
  long long unsigned int a, d;
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
  c_syn0.set_size(size, P, words);
  vocab = (char *)malloc(words * max_w * sizeof(char));
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
    for(int p  = 0; p < P; p++) {
            for (int s = 0; s < size; s++) {
                fread(&c_syn0(s, p, b), sizeof(float), 1, f);
          }
      }
    c_syn0.slice(b) = arma::normalise(c_syn0.slice(b));
    arma::fmat K  = c_syn0.slice(b);
  }
  fclose(f);
  TCN = 0;
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 100000;
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
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    TQ++;
    if (b1 == words) continue;
    if (b2 == words) continue;
    if (b3 == words) continue;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st4)) break;
    if (b == words) continue;
    TQS++;
    float L = 0.0;
    //Find the tangent vector T_A
    arma::fmat T_A = log_map(c_syn0.slice(b1), c_syn0.slice(b2), L);
    //Transport tangent vector T_A to T_C along the geodesic connecting a to c
    arma::fmat T_C = parallel(c_syn0.slice(b1), c_syn0.slice(b3), T_A, L);
    //Get the target matrix (the required word)
    arma::fmat target = exp_map(c_syn0.slice(b3), T_C, L);
    for (c = 0; c < words; c++) {
      if (c == b1) continue;
      if (c == b2) continue;
      if (c == b3) continue;
      dist = 0;
      dist = getNaturalDist(target, c_syn0.slice(c));
      //dist = getChordalDist(target, c_syn0.slice(c)); 
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