#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include "linenoise.h"
#include "vec.h"
#include <algorithm>

#define max_size 2000
#define N 40
#define max_w 50
#define max(x, y) ((x) > (y) ? (x) : (y))

Vec *M;
char *vocab;
long long words, size;
char *bestw[N];
real *quadform;
real *Ay;
real *normalizationFactorR;
real *normalizationFactorL;

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


real getNormalizationFactorL(Vec &v) {
    // this . dot(other)
    float maxdot = 0;
    for(int i = 0; i < N; ++i) {
        const float dist = mulQuadForm(size, v.v, quadform, M[i].v,  Ay, nullptr);
        maxdot = std::max<float>(maxdot, fabs(dist));
    }
    return maxdot;
}
void buildNormalizationFactorLCache() {
    printf("building L cache...\n");
    normalizationFactorL = (real *)malloc(sizeof(real) * words);
    for(int i = 0; i < words; ++ i) {
        printf("\r%20d / %d", i, words);
        fflush(stdout);
        normalizationFactorL[i] = getNormalizationFactorL(M[i]);
    }
    printf("\n");
}

real getNormalizationFactorR(Vec &v) {
    // others . dot (this)
    float maxdot = 0;
    for (int i = 0; i < words; ++i) {
        const float dist = mulQuadForm(size, M[i].v, quadform, v.v,  Ay, nullptr);
        maxdot = max(maxdot, fabs(dist));
    }
    return maxdot;
}

real getNormalizationFactorR(int w) {
    // others . dot (this)
    float maxdot = 0;
    for(int i = 0; i < N; ++i) {

        const float dist = mulQuadForm(size, M[i].v, quadform, M[w].v,  Ay, nullptr);
        maxdot = max(maxdot, fabs(dist));
    }
    return maxdot;
}


void buildNormalizationFactorRCache() {
    printf("building R cache...\n");
    normalizationFactorR = (real *)malloc(sizeof(real) * words);
    printf("\rallocated R cache...\n");
    for(int i = 0; i < words; ++i) {
        printf("\r%d / %d", i, words);
        fflush(stdout);
        // normalizationFactorR[i] = getNormalizationFactorR(M[i]);
        normalizationFactorR[i] = getNormalizationFactorR(i);
    }
    printf("\n");
}

void cosine(Vec vec) {
    real vals[words];
    float dist, len, bestd[N];

    
    {
        printf(
                "\n                                              Word       "
                "Cosine "
                "distance\n----------------------------------------------------"
                "----"
                "----------------\n");
        // for (a = 0; a < size; a++) vec[a] = 0;

        // for (b = 0; b < cn; b++) {
        //     if (bi[b] == -1) continue;
        //     vec.accumscaleadd(1.0, M[bi[b]]);
        //     // for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
        // }
        for (int i = 0; i < std::min<int>(size, 10); i++) {
            printf("%3.2f  ", vec.ix(i));
        }
        printf("\n");
        real len = 0;

        // get the length of largest dot with bi.
        const float vecnorm = getNormalizationFactorL(vec);
        printf("%s:%d\n", __FILE__, __LINE__);

        // for (a = 0; a < size; a++) len += vec[a] * vec[a];
        // len = sqrt(len);
        // for (a = 0; a < size; a++) vec[a] /= len;
        // vec.normalize();
        for (int a = 0; a < N; a++) bestd[a] = -1;
        for (int a = 0; a < N; a++) bestw[a][0] = 0;
        for (int c = 0; c < words; c++) {
            // dist = 0;
            // for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];
            // dist = vec.dotContainmentConstrained(M[c],  0, 2, 0, 2, nullptr, nullptr);
            const float curnorm = normalizationFactorR[c];
            dist = mulQuadForm(size, vec.v, quadform, M[c].v,   Ay, nullptr);
            dist /= vecnorm; 
            dist /= curnorm;
            vals[c] = dist;

            for (int a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (int d = N - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        printf("%s:%d\n", __FILE__, __LINE__);
        for (int a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
        plotHistogram("distances", vals, words, 10);
    }


    {
        printf(
                "\n                                              Word       "
                "Cosine "
                "distance\n----------------------------------------------------"
                "----"
                "----------------\n");

        // get the length of largest dot with bi.
        const float vecnorm = getNormalizationFactorR(vec);
        len = 0;
        // for (a = 0; a < size; a++) len += vec[a] * vec[a];
        // len = sqrt(len);
        // for (a = 0; a < size; a++) vec[a] /= len;
        // vec.normalize();
        for (int a = 0; a < N; a++) bestd[a] = -1;
        for (int a = 0; a < N; a++) bestw[a][0] = 0;
        for (int c = 0; c < words; c++) {
            const float curnorm = normalizationFactorL[c];
            dist = mulQuadForm(size, M[c].v, quadform, vec.v,   Ay, nullptr);
            dist /= curnorm;
            dist /= vecnorm;
            vals[c] = dist;

            for (int a = 0; a < N; a++) {
                if (dist > bestd[a]) {
                    for (int d = N - 1; d > a; d--) {
                        bestd[d] = bestd[d - 1];
                        strcpy(bestw[d], bestw[d - 1]);
                    }
                    bestd[a] = dist;
                    strcpy(bestw[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        for (int a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);

        plotHistogram("distances", vals, words, 10);
    }
}

enum class ASTTy { List, AtomString, Null };

struct AST {
   private:
    std::vector<AST> list_;
    std::string s_;
    ASTTy ty_;

   public:
    AST() : ty_(ASTTy::Null){};
    AST(std::string s) : s_(s), ty_(ASTTy::AtomString) {}
    AST(std::vector<AST> list) : list_(list), ty_(ASTTy::List) {}

    using iterator = std::vector<AST>::iterator;

    void print() {
        switch (ty_) {
            case ASTTy::AtomString:
                std::cout << s_;
                return;
            case ASTTy::List:
                std::cout << '(';
                for (auto it : list_) {
                    it.print();
                    std::cout << ' ';
                }
                std::cout << ')';
                return;
            case ASTTy::Null:
                std::cout << "null";
        }
    }

    ASTTy ty() { return ty_; };
    std::string s() { return s_; };

    AST at(int i) {
        assert(ty_ == ASTTy::List);
        return list_[i];
    }

    int size() {
        assert(ty_ == ASTTy::List);
        return list_.size();
    }
};

std::tuple<AST, char *> parse_(char *str) {

    // consume whitespace after word
    while (str[0] == ' ' && str[0] != 0) str++;

    if (str[0] == 0) return std::make_tuple(AST(), str);



    // if it's not an open brace, it's a word.
    if (str[0] != '(') {
        int offset = 0;
        while (str[offset] != ' ' && str[offset] != ')' && str[offset] != 0)
            offset++;

        // consume whitespace after word
        while (str[0] == ' ' && str[0] != 0) str++;

        return std::make_tuple(AST(std::string(str, offset)), str + offset);
    }

    assert(str[0] == '(');

    // consume the '('
    str++;

    std::vector<AST> list;
    while (str[0] != ')' && str[0] != 0) {
        // consume whitespace
        while (str[0] == ' ' && str[0] != 0) str++;

        AST ast;
        std::tie(ast, str) = parse_(str);

        list.push_back(ast);

        // consume whitespace
        while (str[0] == ' ' && str[0] != 0) str++;
    }

    // consume the ')'
    str++;

    return std::make_tuple(AST(list), str);
};

Vec clone(Vec v) {
    Vec w;
    w.alloc(v.len);
    for (int i = 0; i < v.len; ++i) w.v[i] = v.v[i];
    return w;
}

AST parse(char *str) { return std::get<0>(parse_(str)); }

std::pair<Vec, bool> interpret(AST ast) {
    std::cout << "interpreting: ";
    ast.print();
    std::cout << std::endl;

    bool b;

    switch (ast.ty()) {
        case ASTTy::AtomString: {
            const std::string s = ast.s();
            for (int i = 0; i < words; i++) {
                if (!strcmp(&vocab[i * max_w], s.c_str()))
                    return std::make_pair(clone(M[i]), true);
            }
            std::cout << "|" << s << "|  unknown.\n";
            return std::make_pair(Vec(), false);
        }

        case ASTTy::List: {
            if (ast.at(0).ty() != ASTTy::AtomString) {
                std::cout << "incorrect command -  head should be command: ";
                ast.print();
                return std::make_pair(Vec(), false);
            }

            const std::string s = ast.at(0).s();
            if (s == "+") {
                Vec v;
                std::tie(v, b) = interpret(ast.at(1));
                if (!b) return std::make_pair(Vec(), b);


                for (int i = 2; i < ast.size(); ++i) {
                    Vec w;
                    std::tie(w, b) = interpret(ast.at(i));
                    if (!b) return std::make_pair(Vec(), b);
                    v.accumscaleadd(1.0, w);
                }
                return std::make_pair(v, true);
            }
            if (s == "-") {
                Vec v;
                std::tie(v, b) = interpret(ast.at(1));
                if (!b) return std::make_pair(Vec(), b);

                if (ast.size() == 2) {
                    v.scale(-1, /*gradient=*/nullptr);
                    return std::make_pair(v, true);
                }

                assert(ast.size() == 3);
                Vec w;
                std::tie(w, b) = interpret(ast.at(2));
                if (!b) return std::make_pair(Vec(), b);

                v.accumscaleadd(-1, w);
                return std::make_pair(v, true);
            }

            if (s == "." || s == "dot") {
                if (ast.size() != 3) {
                    std::cout << "Dot needs 2 arguments\n";
                }

                Vec v, w;
                std::tie(v, b) = interpret(ast.at(1));
                if (!b) return std::make_pair(Vec(), b);
                std::tie(w, b) = interpret(ast.at(2));
                if (!b) return std::make_pair(Vec(), b);

                const float d = mulQuadForm(size, v.v, quadform, w.v,  Ay, nullptr);
                std::cout << "rawdot: " << d
                    << "\ndotnorm: " << d / (getNormalizationFactorL(v) * getNormalizationFactorR(w))
                    << "\n";
                return std::make_pair(Vec(), false);
            }

            // left projection.
            if (s == "<." || s == ".<" || s == "lproject" || s == "projectl") {
                if (ast.size() != 4) {
                    std::cout << "left projection needs 3 arguments\n";
                    // p : q :: x : y?
                    // q - p = y - x
                    // y = b - a + x
                    Vec p, q, x;
                    std::tie(p, b) = interpret(ast.at(1));
                    if (!b) return std::make_pair(Vec(), b);
                    std::tie(q, b) = interpret(ast.at(2));
                    if (!b) return std::make_pair(Vec(), b);
                    std::tie(x, b) = interpret(ast.at(3));
                    if (!b) return std::make_pair(Vec(), b);
                }

                
            }

            return interpret(ast.at(0));
        }

        case ASTTy::Null:
            // assert(false && "cannot interpret null ast");
            return std::make_pair(Vec(), false);
    }
}

// completions for linenoise
void completion(const char *buf, linenoiseCompletions *lc) {
    for (int i = 0; i < words; ++i) {
        // TODO: change it so it works when typing stuff. That is,
        // tokenize the string and the decide what completion to add...
        if (strstr(&vocab[i * max_w], buf) == &vocab[i * max_w]) {
            linenoiseAddCompletion(lc, &vocab[i * max_w]);
        }
    }
}

// plot dimension usage
void dimension_usage() {
    double *f = (double *)malloc(size * sizeof(double));
    double *fnorm = (double *)malloc(size * sizeof(double));

    for (int i = 0; i < size; i++) {
        f[i] = 0;
        fnorm[i] = 0;
    }
    for (int w = 0; w < words; w++) {
        for (int i = 0; i < size; i++) {
            const float cur =  M[w].ix(i);
            f[i] += fabs(cur);
        }
    }

    double total = 0;
    for (int i = 0; i < size; ++i) total += f[i];

    // normalize
    for (int i = 0; i < size; ++i) fnorm[i] = (f[i] * 100.0) / total;

    printf("dimension weights as percentage [0..n]:\n");
    for (int i = 0; i < size; ++i) printf("%d: %5.8f\n", i, fnorm[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    char file_name[512];
    FILE *f;

    if (argc != 2) {
        printf("usage: %s <path-to-model.bin>\n", argv[0]);
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
    for (int a = 0; a < N; a++)
        bestw[a] = (char *)malloc(max_size * sizeof(char));

    M = (Vec *)malloc((long long)words * sizeof(Vec));
    printf("setting up quadform...\n");
    quadform = (float*)malloc(sizeof(float) * size * size);
    setupDotContainmentMat(size, quadform);
    printf("setting up Ay...\n");
    Ay = (float *)malloc(sizeof(float) * size * size);

    Vec vec;
    vec.alloc(size);
    // (long long)size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }
    for (int b = 0; b < words; b++) {
        int a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        vocab[b * max_w + a] = 0;
        M[b].alloc(size);
        readvec(f, M[b]);
        printf("%s:", vocab + b * max_w);
        for (int i = 0; i < std::min<int>(size, 10); i++) {
            printf("%3.4f ", M[b].ix(i));
        }
        printf("\n");
        // for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1,
        // f); len = 0; for (a = 0; a < size; a++) len += M[a + b * size] * M[a
        // + b * size]; len = sqrt(len); for (a = 0; a < size; a++) M[a + b *
        // size] /= len;
    }


    fclose(f);

    buildNormalizationFactorLCache();
    buildNormalizationFactorRCache();

    // printf("HACK: CLEARNING 0th and LAST DIMENSION\n");
    // for(int i = 0; i < words; i++) {
    //     M[i].v[0] = M[i].v[size - 1] = 0;
    // }

    dimension_usage();

    linenoiseHistorySetMaxLen(10000);
    linenoiseSetCompletionCallback(completion);
    while (1) {
        char *s = linenoise(">");
        AST ast = parse(s);
        linenoiseFree(s);
        std::cout << "ast: ";
        ast.print();
        std::cout << "\n\n";

        if (ast.ty() == ASTTy::Null) continue;

        Vec v; bool success;
        std::tie(v, success) = interpret(ast);
        if (!success) continue;

        printvec(v, "vector: ", nullptr);
        cosine(v);
        v.freemem();
    }
}
