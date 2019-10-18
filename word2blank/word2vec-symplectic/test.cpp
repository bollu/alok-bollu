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

int main(int argc, char **argv) {
    static const int SEED = 2;
    srand(SEED);

    float vectors[4][4];
    vectors[0][0] = 1;
    vectors[0][1] = 0;
    vectors[0][2] = 0;
    vectors[0][3] = 0;

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

    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            printf("%d %d %4.3f\n", i, j, dotSymplectic(4, vectors[i], vectors[j]));
        }
    }

    return 0;
}
