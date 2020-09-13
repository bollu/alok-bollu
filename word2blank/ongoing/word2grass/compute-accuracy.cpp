#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <armadillo>
#include <iostream>
//compute-accuracy /path/to/model.bin < questions-words.txt > output-file.txt
using namespace std;
using namespace arma;

const long long max_size = 2000;         // max length of strings
const long long N = 1;                   // number of closest words
const long long max_w = 50;              // max length of vocabulary entries

double distance(arma::mat M1, arma::mat M2) {
    // assert (w1 >= 0);
    // assert (w2 >= 0);
    // arma::mat M1(P, size); M1.zeros();
    // arma::mat M2(P, size); M2.zeros();
    // for ( row = 0; row < P; row++) for ( col = 0; col < size; col++) M1(row,col) = M[(size*P*w1) + (row*size) + col]; 
    // for ( row = 0; row < P; row++) for ( col = 0; col < size; col++) M2(row,col) = M[(size*P*w2) + (row*size) + col]; 
    arma::Mat<double> Proj = M1 - M2;
    arma::Mat<double> K = Proj*arma::trans(Proj);
    double distance = sqrt(arma::trace(K)/2);
    return distance ;

    // arma::Mat<double> K = M1*arma::trans(M2);
    // double determinant = arma::det(K);
    // double distance = 1 - (determinant*determinant);
    // return distance;
}

void exp(arma::mat X)
{

}

int main(int argc, char **argv)
{
  FILE *f;
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[N][max_size], file_name[max_size];
  double dist, bestd[N];
  long long words, size, P,  b, c, b1, b2, b3, threshold = 0,row,col;
  long long unsigned int a, d;
  double *M,*T_0;
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
  fscanf(f, "%lld", &P);
  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (double *)malloc(words * P * size * sizeof(double));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * P * size * sizeof(double) / 1048576);
    return -1;
  }
  mat Mat_b1(P,size); Mat_b1.zeros();
  mat Mat_b2(P,size); Mat_b2.zeros();
  mat Mat_b3(P,size); Mat_b3.zeros();
  mat Mat_b4(P,size); Mat_b4.zeros();
  T_0 = (double *)malloc(size * size * sizeof(double));
  if (T_0 == NULL) {
    printf("Cannot allocate memory: %lld MB\n", size * size * sizeof(double) / 1048576);
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
    for (row = 0; row < P; row++) for (col = 0; col < size; col++) fread(&M[col + (row * size) + (b * size * P)], sizeof(double), 1, f);
  }
  fclose(f);
  TCN = 0;
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 20001;
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
    for (a = 0; a < N; a++) bestd[a] = 2000001;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    TQ++;
    if (b1 == words) continue;
    if (b2 == words) continue;
    if (b3 == words) continue;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st4)) break;
    if (b == words) continue;
    for (row = 0; row < P; row++) for (col = 0; col < size; col++) 
    {
      Mat_b2(row, col) = M[col + (row*size) + (b2*size*P)] ;
      Mat_b1(row, col) = M[col + (row*size) + (b1*size*P)] ;
      Mat_b3(row, col) = M[col + (row*size) + (b3*size*P)] ;
    }
    // uword r = arma::rank(Mat.t());
    // fprintf(stderr,"Rank of the matrix is %llu\n", r); 
    // mat Q_Mat;
    // mat R_Mat;
    // arma::qr(Q_Mat, R_Mat, Mat.t());
    // for (row = 0; row < P; row++) for (col = 0; col < size; col++) Mat(row, col) = Q_Mat(col, row); 
    //------------
    TQS++;
    arma::mat B = arma::expmat_sym(arma::logmat_sympd(Mat_b3.t()*Mat_b3) + arma::logmat_sympd(Mat_b2.t()*Mat_b2) - arma::logmat_sympd(Mat_b1.t()*Mat_b1));
    B = arma::orth(B);
    cout << B << endl;
    for (c = 0; c < words; c++) {
      if (c == b1) continue;
      if (c == b2) continue;
      if (c == b3) continue;
      dist = 0;
      for (row = 0; row < P; row++) for (col = 0; col < size; col++) Mat_b4(row, col) = M[col + (row*size) +(c*size*P)];
      // vec s_b1 = arma::svd(Mat_b4 * Mat_b1.t());
      // vec s_b2 = arma::svd(Mat_b4 * Mat_b2.t());
      // vec s_b3 = arma::svd(Mat_b4 * Mat_b3.t());
      // dist = arma::accu(s_b2 % s_b2)*arma::accu(s_b3 % s_b3)/arma::accu(s_b1 % s_b1);
      //dist = distance(B,Mat_b4.t()*Mat_b4);
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
