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
            printf("%20s ∈ %20s %20.3f\n", name.c_str(), name2.c_str(),
                   it.second.dotContainment(it2.second, nullptr, nullptr));
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
                it.second.dotContainment(it2.second, gs[name], gs[name2]);
            printf("----\n");
            printf("%8s ∈ %8s %8.2f\n", name.c_str(), name2.c_str(), d);
            printvec(it.second, name.c_str(), gs[name]);
            printvec(it2.second, name2.c_str(), gs[name2]);
        }
    }
}

void learn3d(int x, bool debug) {
    static const int NDIMS = 3;

    static const real LEARNINGRATE = 0.001;
    Vec normal;
    Vec normaldual;
    normal.alloczero(1 << NDIMS);
    normaldual.alloczero(1 << NDIMS);
    // we do not allocate the scalar part and the final full dimension on
    // purpose. these are degenerate
    for (int i = 0; i < (1 << NDIMS) - 1; ++i)
        normal.v[i + 1] = bool(x & (1 << i));
    normal.normalize();

    // fill the vector with th hodge dual
    normal.hodgedual(normaldual);
    normaldual.accumscaleadd(
        -1.0 * normal.dotContainment(normaldual, nullptr, nullptr), normal);

    Vec random, randomOrig;
    random.alloczero(8);
    randomOrig.alloczero(8);
    for (int i = 1; i < 7; ++i)
        random.v[i] = randomOrig.v[i] =
            0.25 + real(((rand() % 10) - 5.0) / 5.0);
    random.normalize();
    randomOrig.normalize();

    real *grad = (float *)malloc((1 << NDIMS) * sizeof(real));
    real *grad2 = (float *)malloc((1 << NDIMS) * sizeof(real));

    float dot = 1.0, dot2 = 1.0;
    int round = 1;
    static const int NROUNDS = 1e6;
    for (; round < NROUNDS; ++round) {
        for (int i = 0; i < (1 << NDIMS) - 1; ++i) grad[i] = 0;
        for (int i = 0; i < (1 << NDIMS) - 1; ++i) grad2[i] = 0;

        // RANDOM . NORMAL
        // dot += normal.dotContainment(random, true, nullptr, grad);
        {
            dot = random.dotContainment(normaldual, grad, nullptr);
            for (int i = 1; i < (1 << NDIMS) - 1; i++) {
                random.v[i] += LEARNINGRATE * dot * grad[i];
            }

            // for (int i = 0; i < (1<<NDIMS) - 1; ++i) grad[i] = 0;
            // dot = random.dotContainment(normal, grad, nullptr);
            // for (int i = 1; i < (1 << NDIMS) - 1; i++) {
            //     random.v[i] += -1.0 * LEARNINGRATE * dot * grad[i];
            // }
        }

        if ((round + 1) % 5 == 0) {
            random.normalize();

            // printf("***round: %d\n", round);
            // printvec(random, "  random (unnormalized)", nullptr);
            // printf("  random lensq (unnormalized): %f\n", random.lensq());
            // printvec(random, "  random (normalized)", nullptr);
            // printf("  random lensq (normalized): %f\n", random.lensq());
        }
        /*

        // NORMAL . RANDOM dot += normal.dotContainment(random, true, nullptr,
        // grad);
        {
            dot2 = normal.dotContainmentConstrained(random, 0, 2, 0, 2,
                                                    nullptr, grad2);

            // train objects of dimension "dim"
            for (int i = 0; i < min(4, pow2(dim) - 1); i++) {
                random.v[i] += -1.0 * LEARNINGRATE * dot2 * grad2[i];
                // add up the absolute difference in the gradient.
                gradDelta += fabs(avggrad2[i] - grad2[i]);

                // exponential decay
                avggrad2[i] = avggrad2[i] * 0.9 + grad2[i] * 0.1;
            }
        }
        */
    }

    // take the other direction of dot product as well?
    // dot += normal.dotContainment(random, true, nullptr, grad);
    printf("\n\nLEARNING 2D: %d\n", x);
    printvec(normal, "normal", nullptr);
    printvec(normaldual, "normal (hodge dual)", nullptr);
    printvec(randomOrig, "random (starting)", nullptr);
    printf("--\n");
    printvec(random, "random", grad);

    printf("normal . *normal: %.5f\n",
           normal.dotContainment(normaldual, nullptr, nullptr));
    printf("random . *normal: %.5f\n",
           random.dotContainment(normaldual, nullptr, nullptr));
    printf("randomOrig . normal: %.5f\n",
           randomOrig.dotContainment(normal, nullptr, nullptr));
    printf("random . normal: %.5f\n",
           random.dotContainment(normal, nullptr, nullptr));
    printf("normal . random: %.5f\n",
           normal.dotContainment(random, nullptr, nullptr));

    // assert(fabs(random.dotContainment(normal, nullptr, nullptr)) < 0.1);
    // assert(fabs(normal.dotContainment(random, false, nullptr, nullptr)) <
    // 0.01); assert(fabs(random.dotContainment(normal, false, nullptr,
    // nullptr)) < 0.01);
}

int main(int argc, char **argv) {
    static const int SEED = 2;
    srand(SEED);
    testdot();
    testgradient();

    for (int i = 0; i < 64; ++i) {
        learn3d(i, false);
        getchar();
    }

    // learn3d(0, 0, 0, 1, true);
    return 1;
}
