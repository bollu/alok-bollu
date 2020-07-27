#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int P = 2, N = 2;
void dgetrf_(int *rows, int *cols, double *matA, int *LDA, int *IPIV, int *INFO);
void dgetri_(int *N, double *matA, int *LDA, int *IPIV, double *WORK, int *LWORK, int *INFO);

//for Q calculation 
void dgeqrf_(int *rows, int *cols, double *matA, int *LDA, double *TAU, double *WORK, int *LWORK, int *INFO);
void dorgqr_(int *rows, int *cols, int *K, double *matA, int *LDA, double *TAU, double *WORK, int *LWORK, int *INFO);
// 1 0
// 0 1
double *C, *X, *M;

void init_var()
{
  int a, b;
  a = posix_memalign((void **)&M, 128, (int)P * P * sizeof(double));
  if (M == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a=0; a<P; a++) for (b=0; b<P; b++)  M[a*P +b] =  rand()%10 + 1;

  a = posix_memalign((void **)&C, 128, (int)P * N * sizeof(double));
  if (C == NULL) {printf("Memory allocation failed\n"); exit(1);}
  C[0] = 0;
  C[1] = 0;
  C[2] = 0;
  C[3] = 0;


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


void learnX()
{
    int b, c;
    //dloss/dX = 2*tr[(M.T M)^-1 (C.T X)]*something
    //calculate the metric
    double *neu1e = (double *)calloc(P*N, sizeof(double));
    for (b = 0; b < P; b++) for (c = 0; c < N; c++) neu1e[b*N + c] = 0;
    double f = 0.0;
    //Calculate M.T M, store it in Denom 
    double Denom[P][P]; 
    memset(Denom, 0, P*P*sizeof(double));
    double elem_sum = 0.0 ;
    for (b = 0; b < P; b++)
    {
        for (c = 0; c < P; c++)
        {
            elem_sum = 0.0 ;
            for ( int k = 0; k < P; k++)
                elem_sum += M[k*P + b]*M[k*P + c];
            Denom[b][c] = elem_sum;
        }      
    }

    int INFO;
    int PP;
    PP = P * P;
    int *pivotArray = malloc(sizeof(int)*P);
    double *lapackWorkspace = malloc(sizeof(double)*PP);
    dgetrf_(&P, &P, Denom[0], &P, pivotArray, &INFO);
    if(INFO !=0){fprintf(stderr,"dgetrf calculation failed, error code %d\n",INFO);exit(1);}

    dgetri_(&P, Denom[0], &P, pivotArray, lapackWorkspace, &PP, &INFO);
    if(INFO !=0){fprintf(stderr,"dgetri calculation failed, error code %d\n",INFO);exit(1);}
    
    //printf("AFTER INVERSE:\n");
    //for (b=0; b<P; b++) { for (c=0; c<P; c++) printf("%f ",Denom[b][c]); printf("\n");} 
    double Numer[P][P]; 
    memset(Numer, 0, P*P*sizeof(double));
    elem_sum = 0.0 ;
    for (b = 0; b < P; b++)
    {
        for (c = 0; c < P; c++)
        {
            elem_sum = 0.0 ;
            for ( int k = 0; k < N; k++)
                elem_sum += C[ b*N + k]*X[c*N + k];
            Numer[b][c] = elem_sum;
        }
        
    }
    //Calculate trace(Denom @ Numer)
    for ( b = 0; b < P; b++) for ( c = 0; c < P; c++) f += Denom[b][c] * Numer[c][b];
    
    for (b = 0; b < P; b++)  
    {  
        for (c = 0; c < N; c++)
        { 
            for ( int d = 0; d < P; d++) neu1e[b*N + c] += Denom[b][d] * C[b*N + c]; 
            neu1e[b*N + c] = X[b*N +c] + (2 * f * 0.001* neu1e[b*N + c]);
        }
    }

    // Get Q factor from QR factorization of NEU2E
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

    for (int i = 0; i < 500; ++i) learnX();

    for(int a=0; a<P; a++) { for(int b=0; b<N; b++) printf("%f ",X[a*N + b]); printf("\n");}
    return 0;
}

