// Run tests on the implementation of geometric algebra and ensure that
// we have implemented this correctly
#include "vec.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int testdot() {
    Vec scalar, x, y, xy;

    scalar.alloczero(4);
    x.alloczero(4);
    y.alloczero(4);
    xy.alloczero(4);

    scalar.v[0] = 11;
    x.v[1] = 3;
    y.v[2] = 5;
    xy.v[3] = 7;


    printvec(scalar, "scalar", nullptr);
    printvec(x, "x", nullptr);
    printvec(y, "y", nullptr);
    printvec(xy, "xy", nullptr);


}

int main(int argc, char **argv) {
    testdot();
}
