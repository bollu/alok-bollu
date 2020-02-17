//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define max_size 2000
#define N 40
#define max_w 50

typedef float real;

FILE *f;
char st1[max_size];
char *bestw[N];
char file_name[max_size], st[100][max_size];
float dist, len, bestd[N], vec[max_size];
long long words, size, a, b, c, d, cn, bi[100];
float *M;
char *vocab;

#define NUM_SPARKS 100

float entropylog(float x) {
    assert(x >= 0);
    if (x < 1e-8) {
        return 0;
    }
    return log(x);
}

float h(float *a) {
    int tot = 0;
    for(int i = 0; i < size; ++i) { tot += a[i] * entropylog(a[i]); }
    tot = -tot;
    assert(tot >= 0);
    return tot;
}

float entropyfuzzy(float *v) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        H += -v[i] * entropylog(v[i]) - (1 - v[i]) * entropylog(1 - v[i]);
    }
    assert(H >= 0);

    return H;
}

// this is fuzzy KL divergence
// int p(x) log (p(x)/q(x)) dx
float kl(float *a, float *b) {
    float out = 0;
#ifdef DEBUG
    float tota = 0; float totb = 0;
#endif
    for(int i = 0; i < size; i++) { 
#ifdef DEBUG
        tota += a[i];
        totb += b[i];
        
        assert(a[i] >= 0); assert(a[i] <= 1);
        assert(b[i] >= 0); assert(b[i] <= 1);
#endif
        out += a[i] * (entropylog(a[i]) - entropylog(b[i]));
    }
#ifdef DEBUG
    // assert(fabs(tota - 1) < 1e-5);
    // assert(fabs(totb - 1) < 1e-5);
    assert(out >= 0);
#endif

    return out;
}


float crossentropy(float *a, float *b) {
    return h(a) + kl(a, b);
}


double klfuzzy(float *v, float *w) {
    double H = 0;
    for(int i = 0; i < size; ++i)  {
        H += -v[i] * entropylog(w[i]) - (1 - v[i]) *  entropylog((1 - w[i]));
    }
    return H;
}

float crossentropyfuzzy(float *v, float *w) {
    float H = 0;
    for(int i = 0; i < size; ++i)  {
        assert(v[i] >= 0); assert(v[i] <= 1);
        assert(w[i] >= 0); assert(w[i] <= 1);
        H += v[i] * (entropylog(v[i]) - entropylog(w[i])) + 
            (1 - v[i]) * (entropylog((1 - v[i])) - entropylog((1-w[i])));
    }
    assert(H >= 0);
    return H;

}

float clamp01(float x) {
    const float EPS = 1e-4;
    if (x >= 1.0 - EPS) { return 1.0 - EPS; }
    else if (x < EPS) { return EPS; }
    else { return x; }
}

/*
>>> x * (sympy.log(x) - sympy.log(y))
x*(log(x) - log(y))
>>> kl = x * (sympy.log(x) - sympy.log(y))
>>> kl.diff(x)
log(x) - log(y) + 1
>>> kl.diff(y)
-x/y
*/
void klgrad_left(float g, float *a, float *b, float *da) {
    for(int i = 0; i < size; ++i) {
        float d = (entropylog(a[i]) - entropylog(b[i])) + 1.0;
        da[i] = clamp01(da[i] + g * d);
    }
}

void klgrad_right(float g, float *a, float *b, float *db) {
    for(int i = 0; i < size; ++i) {
        float d = (-1.0 * a[i] / b[i]);
        db[i] = clamp01(db[i] + g * d);
    }
}

void normalize(float *a) {
    float tot = 0;
    for(int i = 0; i < size; ++i) { tot += a[i]; }
    for(int i = 0; i < size; ++i) { a[i] /= tot; }

}



void plotHistogram(const char *name, real *vals, int n, int nbuckets) {
    // number of values in each bucket.
    int buckets[nbuckets];
    for(int i = 0; i < nbuckets; ++i) buckets[i] = 0;

    real vmax = vals[0];
    real vmin = vals[0];
    for(int i = 0; i < n; ++i) vmax = vals[i] > vmax ? vals[i] : vmax;
    for(int i = 0; i < n; ++i) vmin = vals[i] < vmin ? vals[i] : vmin;

    real multiple = (vmax - vmin) / nbuckets;

    for(int i = 0; i < n; ++i) {
        int b = floor((vals[i] - vmin) / multiple);
        b = b >= nbuckets ? (nbuckets -1): (b < 0 ? 0 : b);
        buckets[b]++;
    }
    
    int total = 0;
    for(int i = 0; i < nbuckets; ++i) total += buckets[i];

    printf("%s: |", name);
    for(int i = 0; i < nbuckets; ++i) {
        printf(" %f ", ((buckets[i] / (real)total)) * 100.0);
    }
    printf("|");

}


// find dot product of two words
void dot() {
    float lensq;
    if (cn != 3) {
        printf("ERROR: expected two vectors to find dot product\n");
    }

    float d = 0;
    for (a = 0; a < size; a++) d += M[a + bi[1] * size] * M[a + bi[2] * size];

    lensq = 0;
    for (a = 0; a < size; a++)
        lensq += M[a + bi[2] * size] * M[a + bi[2] * size];
    d /= sqrt(lensq);

    lensq = 0;
    for (a = 0; a < size; a++)
        lensq += M[a + bi[1] * size] * M[a + bi[1] * size];
    d /= sqrt(lensq);
    lensq = 0;

    printf("dot: %f\n", d);
}

real *vals; // [words];

void cosine() {
    for(int i = 0; i < words; ++i) vals[i] = 0;

    printf(
        "\n                                              Word       "
        "Cosine "
        "distance\n----------------------------------------------------"
        "----"
        "----------------\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {
        if (bi[b] == -1) continue;
        for (a = 0; a < size; a++) vec[a] = M[a + bi[b] * size];
    }
    // len = 0;
    // for (a = 0; a < size; a++) len += vec[a] * vec[a];
    // len = sqrt(len);
    // for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = 100;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
        a = 0;
        for (b = 0; b < cn; b++)
            if (bi[b] == c) a = 1;
        if (a == 1) continue;
        dist = 0;
        for (a = 0; a < size; a++) { 
            // dist = klfuzzy(vec, M + c*size) + entropyfuzzy(M + c*size);
            // dist = klfuzzy(M + c * size, vec); 
            dist = klfuzzy(vec, M + c * size);
            assert(dist >= 0);
            //dist += vec[a] * M[a + c * size];
        }

        // store the distance value
        vals[c] = dist;

        for (a = 0; a < N; a++) {
            if (dist < bestd[a]) {
                for (d = N - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], &vocab[c * max_w]);
                break;
            }
        }
    }
    for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);

    real vals2[100];
    for(int i = 0; i < 100; ++i) {
        vals2[i] = rand()  % 10;
    }
    plotHistogram("distances", vals, words, 10);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf(
            "Usage: ./distance <FILE>\nwhere FILE contains word projections in "
            "the BINARY FORMAT\n");
        return 0;
    }
    strcpy(file_name, argv[1]);
    f = fopen(file_name, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (float *)malloc((long long)words * (long long)size * sizeof(float));
    vals = (float *)malloc((long long)words * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }
    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        vocab[b * max_w + a] = 0;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        printf("%20s : %10.7f\n", vocab + b * max_w, entropyfuzzy(M + b*size));
        // normalize(M + b * size);

        // len = 0;
        // for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        // len = sqrt(len);
        // for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    fclose(f);
    while (1) {
        for (a = 0; a < N; a++) bestd[a] = 0;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        printf("Enter word or sentence (EXIT to break): ");
        a = 0;
        while (1) {
            st1[a] = fgetc(stdin);
            if ((st1[a] == '\n') || (a >= max_size - 1)) {
                st1[a] = 0;
                break;
            }
            a++;
        }
        if (!strcmp(st1, "EXIT")) break;
        cn = 0;
        b = 0;
        c = 0;
        while (1) {
            st[cn][b] = st1[c];
            b++;
            c++;
            st[cn][b] = 0;
            if (st1[c] == 0) break;
            if (st1[c] == ' ') {
                cn++;
                b = 0;
                c++;
            }
        }
        cn++;
        for (a = 0; a < cn; a++) {
            for (b = 0; b < words; b++)
                if (!strcmp(&vocab[b * max_w], st[a])) break;
            if (b == words) b = -1;
            bi[a] = b;
            printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
            if (b == -1 && strcmp(st[0], "DOT") != 0) {
                printf("Out of dictionary word!\n");
                break;
            }
        }
        if (b == -1 && strcmp(st[0], "DOT") != 0) continue;

        if (!strcmp(st[0], "DOT")) {
            dot();
        } else {
            cosine();
        }
    }
    return 0;
}
