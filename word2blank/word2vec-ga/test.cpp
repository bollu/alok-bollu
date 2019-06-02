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

void testdot() {
    std::map<std::string, Vec> vs;

    vs["scalar"].alloczero(4);
    vs["x"].alloczero(4);
    vs["y"].alloczero(4);
    vs["xy"].alloczero(4);

    vs["scalar"].v[0] = 11;
    vs["x"].v[1] = 3;
    vs["y"].v[2] = 5;
    vs["xy"].v[3] = 7;

    for (auto it : vs) {
        std::string name = it.first;
        printvec(it.second, name.c_str(), nullptr);
        printf("\n");
    }

    printf("***dot products***\n");
    for (auto it : vs) {
        const std::string name = it.first;
        for (auto it2 : vs) {
            const std::string name2 = it2.first;
            printf(
                "%8s %8s %10.3f\n", name.c_str(), name2.c_str(),
                it.second.dotContainment(it2.second, false, nullptr, nullptr));
        }
    }
}

int main(int argc, char **argv) {
    testdot();
    return 1;
}
