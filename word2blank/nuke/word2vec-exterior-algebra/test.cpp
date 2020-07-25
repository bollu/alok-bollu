// Run tests on the implementation of geometric algebra and ensure that
// we have implemented this correctly
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include "vec.h"

void printRandDot() {
    static const int N = 10000;
    float v[2][N];

    for (int n = 0; n < 100; n++) {
        for(int i = 0; i < 2; ++i) {
            for(int j = 0; j < N; ++j) {
                v[i][j] =  ((real)(rand() % 2 == 0 ? 1 : -1) * (rand() % 1000) / 1000.0) / (float)N;
                printf("%4.8f ", v[i][j]);
            }
            printf("\n");
        }

        printf("%4.2f\n", dotSymplectic(N, v[0], v[1]));
    }
}

void printdot() {
    static const int N = 4;
    // (x0 y2 - y0 x2) + (x1 y3 - x3 y1)
    // dx0 = y2
    // dx1 = y3
    // dx2 = -y0
    // dx3 = -y1
    //
    // dy0 = x2
    // dy1 = -x3
    // dy2 = -x0
    // dy3 = x1


    float vectors[N][N];
    float grad[N];
    vectors[0][0] = 1;
    vectors[0][1] = 0;
    vectors[0][2] = 0;
    vectors[0][3] = 0;

    // 0 . 0 -> 

    vectors[1][0] = 0;
    vectors[1][1] = 1;
    vectors[1][2] = 0;
    vectors[1][3] = 0;


    vectors[2][0] = 0;
    vectors[2][1] = 0;
    vectors[2][2] = 1;
    vectors[2][3] = 0;


    vectors[3][0] = 0;
    vectors[3][1] = 0;
    vectors[3][2] = 0;
    vectors[3][3] = 1;

    for(int i = 0; i < 1; ++i) {
        for(int j = 0; j < N; ++j) {
            printf("---\n");
            printf("%d %d | dot: %4.3f\n", i, j, dotSymplectic(N, vectors[i], vectors[j]));
            
            gradLeftSymplectic(N, vectors[j], grad);
            printf("gradleft: ");
            for(int i = 0; i < N; ++i) printf("%4.2f ", grad[i]);

            for(int i = 0; i < N; ++i) grad[i] = 0;

            gradRightSymplectic(N, vectors[i], grad);
            printf("gradright: ");
            for(int i = 0; i < N; ++i) printf("%4.2f ", grad[i]);
            printf("\n");
        }
    }
}

void traindot(float target) {
    static const int N = 2;

    float v[2][N];
    float gv[2][N];

    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < N; ++j) {
            v[i][j] = (rand() % 2 ? 1 : -1) * (rand() % 1000) / 1000.0;
        }
    }

    for(int i = 0; i < 10000; ++i) {
        float d = dotSymplectic(N, v[0], v[1]);
        float loss = target - d;
        if (loss * loss < 1e-4) return;

        printf("%d\td: %7.3f\tloss: %4.2f", i, d, loss);

        for(int j = 0; j < 2; ++j) {
            for(int k = 0; k < N; ++k) {
                gv[j][k] = 0;
            }
        }

        gradLeftSymplectic(N, v[1], gv[0]);
        gradLeftSymplectic(N, v[0], gv[1]);

        for(int j = 0; j < 2; ++j) {
            for(int k = 0; k < N; ++k) {
                v[j][k] += loss * gv[j][k] * 1e-1;
            }
        }

        for(int j = 0; j < 2; ++j) {
            printf("\tv(%d): ", j);
            for(int k = 0; k < N; ++k) {
                printf("%4.1f ", v[j][k]);
            }

            printf("\tgv(%d): ", j);
            for(int k = 0; k < N; ++k) {
                printf("%4.1f ", gv[j][k]);
            }
        }
        printf("\n");



    }
}
int main(int argc, char **argv) {
    static const int SEED = 2;
    srand(SEED);


    // printdot();
    traindot(1.0);
    printRandDot();


    return 0;
}
