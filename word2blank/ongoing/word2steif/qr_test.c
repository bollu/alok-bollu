#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<lapacke.h>

#define LAPACK_COL_MAJOR  102
//void dgeqrf_(long long *rows, long long *cols, double *matA, long long *LDA, double *TAU, double *WORK, long long *LWORK, int *INFO);
//void dorgqr_(long long *rows, long long *cols, long long *K, double *matA, long long *LDA, double *TAU, double *WORK, long long *LWORK, int *INFO);
int main()
{
    long long i;
    long long N=10;
    long long P=2;
    long long K=P; 
    double *matA=malloc(sizeof(double)*P*N);

    matA[0*N + 0]=0.037958;
    matA[0*N + 1]=-0.009462;
    matA[0*N + 2]=0.047568;
    matA[0*N + 3]=-0.026036;
    matA[0*N + 4]=-0.041821;
    matA[0*N + 5]=0.003996;
    matA[0*N + 6]=0.036392; 
    matA[0*N + 7]=0.040205;
    matA[0*N + 8]=-0.024628;
    matA[0*N + 9]=0.037474;
    matA[1*N + 0]= -0.042654;
    matA[1*N + 1]= -0.041502;
    matA[1*N + 2]=0.016101;
    matA[1*N + 3]= -0.013716;
    matA[1*N + 4]= -0.000510; 
    matA[1*N + 5]= 0.036653;
    matA[1*N + 6]= 0.031927;
    matA[1*N + 7]= -0.029262;
    matA[1*N + 8]= -0.023111;
    matA[1*N + 9]= 0.007484;
    
    long long LDA=N;
    int INFO;

//lapack_int LAPACKE_dgeqrf( int matrix_layout, lapack_int m, lapack_int n, double* a, lapack_int lda, double* tau );
//lapack_int LAPACKE_dorgqr( int matrix_layout, lapack_int m, lapack_int n, lapack_int k, double *a, lapack_int lda, double * tau);

    long long int LWORK2=P, K2 = P, LDA2=N;
    double *TAU2=malloc(sizeof(double)*K2);
    // perform the QR factorization
    INFO = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, N, P, matA, LDA2, TAU2);
    if(INFO !=0) {fprintf(stderr,"QR factorization of Syn0 failed, error code %d\n",INFO);exit(1);}
    INFO = LAPACKE_dorgqr(LAPACK_COL_MAJOR, N, P, K2, matA, LDA2, TAU2);
    if(INFO !=0) {fprintf(stderr,"QR factorization of Syn0 failed, error code %d\n",INFO);exit(1);}

    // double *TAU=malloc(sizeof(double)*K);
    // long long LWORK=P;
    // double *WORK=malloc(sizeof(double)*LWORK);
    // // perform the QR factorization
    // dgeqrf_(&N, &P, matA, &LDA, TAU, WORK, &LWORK, &INFO);
    
    // double *WORK1=malloc(sizeof(double)*LWORK);
    // dorgqr_(&N, &P, &K, matA, &LDA, TAU, WORK1, &LWORK, &INFO);
    // if(INFO !=0){fprintf(stderr,"QR factorization failed, error code %d\n",INFO);exit(1);}

    for(i=0; i<P; i++)
    {
        for(long long j=0; j<N; j++)
            printf("%f ",matA[i*N+j]);
    printf("\n");
    }

    free(TAU2);
    free(matA);
    return 0;
}
