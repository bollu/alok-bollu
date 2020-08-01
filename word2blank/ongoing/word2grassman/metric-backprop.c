#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int P = 2, N = 3;

//for Q calculation 
void dgeqrf_(int *rows, int *cols, double *matA, int *LDA, double *TAU, double *WORK, int *LWORK, int *INFO);
void dorgqr_(int *rows, int *cols, int *K, double *matA, int *LDA, double *TAU, double *WORK, int *LWORK, int *INFO);

//for svd calculation
void dgesdd(char* jobz, int* rows, int* cols, double* matA, int* LDA, double* S, double* U, int* LDU, double* Vt, int* LDVt,
        double* WORK, int* LWORK, int* IWORK, int* INFO );

// 1 0
// 0 1
// 0 0  
double *C, *X;

void init_var()
{
  int a, b;

  a = posix_memalign((void **)&C, 128, (int)P * N * sizeof(double));
  if (C == NULL) {printf("Memory allocation failed\n"); exit(1);}
  C[0] = 1;
  C[1] = 0;
  C[2] = 0;
  C[3] = 0;
  C[4] = 1;
  C[5] = 0;
  
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&X, 128, (int)P * N* sizeof(double));
  if (X == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a=0; a<P; a++) for (b=0; b<N; b++) {
    // rnext = r * CONST + CONST2
    // 0... 2^32 - 1
    next_random = next_random * (unsigned long long)25214903917 + 11;
    
    X[a*N + b ] = (((next_random & 0xFFFF) / (double)65536) - 0.5) / N;
  }
}


void learnX(double *QC)
{
    int b, c;
    //LOSS = \sum_i theta_i*theta_i
    //calculate the metric
    double *neu1e = (double *)calloc(P*N, sizeof(double));
    for (b = 0; b < P; b++) for (c = 0; c < N; c++) neu1e[b*N + c] = 0;
    double f = 0.0;
    
    //

    //Get Q factor from QR factorization of NEU2E
    int LWORK2=P, K2 = P, LDA2=N;
    double *TAU2=malloc(sizeof(double)*K2);
    double *WORK2=malloc(sizeof(double)*LWORK2);
    // perform the QR factorization
    dgeqrf_(&N, &P, neu1e, &LDA2, TAU2, WORK2, &LWORK2, &INFO);
    if(INFO !=0) {fprintf(stderr,"QR factorization of Syn0 failed, error code %d\n",INFO);exit(1);}
    double *WORK3=malloc(sizeof(double)*LWORK2);
    dorgqr_(&N, &P, &K2, neu1e, &LDA2, TAU2, WORK3, &LWORK2, &INFO);
    if(INFO !=0) {fprintf(stderr,"QR factorization of Syn0 failed, error code %d\n",INFO);exit(1);}

    free(TAU2);
    free(WORK2);
    free(WORK3);

    // UPDATE GRADIENT OF SYN1NEG (CONTEXT)
        for (b = 0; b < P; b++) { for (c = 0; c < N; c++) { for (int k=0; k<P; k++)
            X[b*N + c] = neu1e[k*N+c]*M[k*P + b]; } }
}


int main(int argc, char **argv) {
    init_var();
    double *neu1e = (double *)calloc(P*N, sizeof(double));
    for (b = 0; b < P; b++) for (c = 0; c < N; c++) neu1e[b*N + c] = C[b*N + c];
    
    int LWORK2=P, K2 = P, LDA2=N;
    double *TAU2=malloc(sizeof(double)*K2);
    double *WORK2=malloc(sizeof(double)*LWORK2);
    // perform the QR factorization
    dgeqrf_(&N, &P, neu1e, &LDA2, TAU2, WORK2, &LWORK2, &INFO);
    if(INFO !=0) {fprintf(stderr,"QR factorization of Syn0 failed, error code %d\n",INFO);exit(1);}
    double *WORK3=malloc(sizeof(double)*LWORK2);
    dorgqr_(&N, &P, &K2, neu1e, &LDA2, TAU2, WORK3, &LWORK2, &INFO);
    if(INFO !=0) {fprintf(stderr,"QR factorization of Syn0 failed, error code %d\n",INFO);exit(1);}

    free(TAU2);
    free(WORK2);
    free(WORK3);

    for (int i = 0; i < 500; ++i) learnX(neu1e);

    for(int a=0; a<P; a++) { for(int b=0; b<N; b++) printf("%f ",X[a*N + b]); printf("\n");}
    return 0;
}

