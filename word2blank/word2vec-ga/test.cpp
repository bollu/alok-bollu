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

void learn2d(int x0, int x1, int x2, int x3) {
    static const real LEARNINGRATE = 0.1;
    Vec normal;
    normal.alloczero(4);
    normal.v[0] = x0;
    normal.v[1] = x1;
    normal.v[2] = x2;
    normal.v[3] = x3;

    Vec random, randomOrig;
    random.alloc(4);
    randomOrig.alloc(4);
    for (int i = 1; i <= 2; ++i)
        random.v[i] = randomOrig.v[i] = real(((rand() % 10) - 5.0) / 5.0);

    real *grad = (float *)malloc(4 * sizeof(real));
    real *grad2 = (float *)malloc(4 * sizeof(real));
    real *avggrad = (float *)calloc(4, sizeof(real));
    real *avggrad2 = (float *)calloc(4, sizeof(real));

    float dot = 1.0, dot2 = 1.0;
    int round = 1;
    int dim = 0;  // current dimension we are optimising;
    static const int NROUNDS = 1000;
    for (; round < NROUNDS; ++round) {
        for (int i = 0; i < 4; ++i) grad[i] = 0;
        for (int i = 0; i < 4; ++i) grad2[i] = 0;

        // Uniformly sample from {00, 01, 10, 11} and then pick the number
        // of dimensions as the number of 1s.
        const int base = (1 << dim) - 1;
        // const int curdim = __builtin_popcount(r);

        // delta between average gradient and new gradient
        float gradDelta = 0.0;

        // RANDOM . NORMAL
        // dot += normal.dotContainment(random, true, nullptr, grad);
        {
            dot = random.dotContainmentConstrained(normal, dim, dim + 1, 0, 2,
                                                    grad, nullptr);

            // train objects of dimension "dim"
            for (int i = base; i < base + C[2][dim]; i++) {
                random.v[i] += -1.0 * LEARNINGRATE * dot * grad[i];

                // add up the absolute difference in the gradient.
                gradDelta += fabs(avggrad[i] - grad[i]);

                // exponential decay
                avggrad[i] = avggrad[i] * 0.9 + grad[i] * 0.1;
            }
        }

        // NORMAL . RANDOM dot += normal.dotContainment(random, true, nullptr, grad);
        {
            dot2 = normal.dotContainmentConstrained(random, 0, 2, dim, dim + 1,
                                                    nullptr, grad2);

            // train objects of dimension "dim"
            for (int i = base; i < base + C[2][dim]; i++) {
                random.v[i] += -1.0 * LEARNINGRATE * dot2 * grad2[i];
                // add up the absolute difference in the gradient.
                gradDelta += fabs(avggrad2[i] - grad2[i]);

                // exponential decay
                avggrad2[i] = avggrad2[i] * 0.9 + grad2[i] * 0.1;
            }
        }

        // for (int i = 0; i < 4; ++i) printf("%7.5f ", random.v[i]);
        // printf(" | dot %7.5f | dot2 %7.5f | dim %5d |round %5d |∇δ: %f\n", dot,
        //        dot2, dim, round, gradDelta);

        if (round >= (1 + dim) * (NROUNDS / 4)) {
            dim++;
            // if (dim > 3) break;
        }
        /*
        if (gradDelta < 0.00001) {
            dim++;
            if (dim > 3) break;
        }
        */
    }

    // take the other direction of dot product as well?
    // dot += normal.dotContainment(random, true, nullptr, grad);
    printf("\n\nLEARNING 2D: (%d %d %d %d)\n", x0, x1, x2, x3);
    // printvec(normal, "normal", grad);
    printvec(randomOrig, "random (starting)", nullptr);
    printf("--\n");
    printvec(random, "random", grad);
    printf("normal . random: %.5f\n",
           normal.dotContainment(random, false, nullptr, nullptr));
    printf("random . normal: %.5f\n",
           random.dotContainment(normal, false, nullptr, nullptr));
}

int main(int argc, char **argv) {
    static const int SEED = 2;
    srand(SEED);
    testdot();
    testgradient();
    learn2d(1, 0, 0, 0);
    learn2d(0, 1, 0, 0);
    learn2d(0, 0, 1, 0);
    learn2d(0, 0, 0, 1);
    learn2d(0, 1, 1, 0);
    return 1;
}
