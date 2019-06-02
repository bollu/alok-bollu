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

std::map<std::string, Vec> standard_vectors() {
    std::map<std::string, Vec> vs;

    vs["11"].alloczero(4);
    vs["3x"].alloczero(4);
    vs["5y"].alloczero(4);
    vs["7xy"].alloczero(4);
    vs["(2x 4y 1)"].alloczero(4);

    vs["11"].v[0] = 11;
    vs["3x"].v[1] = 3;
    vs["5y"].v[2] = 5;
    vs["7xy"].v[3] = 7;
    vs["(2x 4y 1)"].v[0] = 1;
    vs["(2x 4y 1)"].v[1] = 2;
    vs["(2x 4y 1)"].v[2] = 4;

    return vs;
}
void testdot() {
    std::map<std::string, Vec> vs = standard_vectors();

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
                "%20s %20s %20.3f\n", name.c_str(), name2.c_str(),
                it.second.dotContainment(it2.second, false, nullptr, nullptr));
        }
    }
}

void testgradient() {
    std::map<std::string, Vec> vs = standard_vectors();
    std::map<std::string, real *> gs;
    for (auto it : vs) {
        gs[it.first] = (real *)calloc(it.second.len, sizeof(real));
    }

    printf("**** dot products with gradient ****");
    for (auto it : vs) {
        const std::string name = it.first;
        for (auto it2 : vs) {
            const std::string name2 = it2.first;

            for (int i = 0; i < it.second.len; ++i) {
                gs[name][i] = 0;
                gs[name2][i] = 0;
            }

            const real d =
                it.second.dotContainment(it2.second, true, gs[name], gs[name2]);
            printf("----\n");
            printf("dot %8s %8s %8.2f\n", name.c_str(), name2.c_str(), d);
            printvec(it.second, name.c_str(), gs[name]);
            printvec(it2.second, name2.c_str(), gs[name2]);
        }
    }
}

void learnline() {
    static const real learning_rate = 0.01;
    printf("LEARNING TO BE NORMAL TO A HYPERPLANE\n");
    Vec normal;
    normal.alloczero(4);
    normal.v[1] = 1;
    normal.v[2] = 2;

    Vec random;
    random.alloc(4);
    for (int i = 0; i < 4; ++i) random.v[i] = ((rand() % 8) - 4) / 4.0;

    printvec(normal, "normal", nullptr);
    printvec(random, "random", nullptr);

    real *grad = (float *)malloc(4 * sizeof(real));

    float dot = 1.0;
    int round = 1;
    for (; fabs(dot) > 0.001 && round < 1000; ++round) {
        for (int i = 0; i < 4; ++i) grad[i] = 0;

        dot = 0;
        // dot += random.dotContainment(normal, true, grad, nullptr);
        // take the other direction of dot product as well?
        dot += normal.dotContainment(random, true, nullptr, grad);
        printf("#%d  ", round);
        printf("dot: %.5f\n", dot);
        printvec(random, "", grad);
        for (int i = 0; i < 4; ++i)
            random.v[i] += -1.0 * learning_rate * dot * grad[i];
    }
}

int main(int argc, char **argv) {
    testdot();
    testgradient();
    learnline();
    return 1;
}
