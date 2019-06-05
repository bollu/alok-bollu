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

#define max_size 2000
#define N 10
#define max_w 50

Vec *M;
char *vocab;
long long words, size;
char *bestw[N];

void cosine(Vec vec) {
    float dist, len, bestd[N];

    printf(
        "\n                                              Word       "
        "Cosine "
        "distance\n----------------------------------------------------"
        "----"
        "----------------\n");
    len = 0;
    // for (a = 0; a < size; a++) len += vec[a] * vec[a];
    // len = sqrt(len);
    // for (a = 0; a < size; a++) vec[a] /= len;
    vec.normalize();
    for (int a = 0; a < N; a++) bestd[a] = -1;
    for (int a = 0; a < N; a++) bestw[a][0] = 0;
    for (int c = 0; c < words; c++) {
        float dist = vec.dotContainment(M[c],
                                        /*grad=*/false, nullptr, nullptr);
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

Vec interpret(AST ast) {
    std::cout << "interpreting: ";
    ast.print();
    std::cout << std::endl;

    switch (ast.ty()) {
        case ASTTy::AtomString: {
            const std::string s = ast.s();
            for (int i = 0; i < words; i++) {
                if (!strcmp(&vocab[i * max_w], s.c_str())) return clone(M[i]);
            }
            std::cout << "|" << s << "|  unknown.\n";
            return Vec();
        }

        case ASTTy::List: {
            if (ast.at(0).ty() != ASTTy::AtomString) {
                std::cout << "incorrect command -  head should be command: ";
                ast.print();
                return Vec();
            }

            const std::string s = ast.at(0).s();
            if (s == "+") {
                Vec v = interpret(ast.at(1));

                for (int i = 2; i < ast.size(); ++i) {
                    Vec w = interpret(ast.at(i));
                    v.accumscaleadd(1.0, w);
                }
                return v;
            }
            if (s == "-") {
                if (ast.size() > 3) {
                    std::cout << "- only takes 1 or 2 parameters.";
                    ast.print();
                    return Vec();
                }

                Vec v = interpret(ast.at(1));

                if (ast.size() == 2) {
                    v.scale(-1, /*gradient=*/nullptr);
                    return v;
                }

                assert(ast.size() == 3);
                Vec w = interpret(ast.at(2));
                v.accumscaleadd(-1, w);
                return v;
            }

            if (s == "." || s == "dot") {
                if (ast.size() != 3) {
                    std::cout << "Dot needs 2 arguments\n";
                }

                Vec v = interpret(ast.at(1));
                Vec w = interpret(ast.at(2));
                std::cout << "dot: "
                          << v.dotContainment(w, /*grad=*/false, nullptr,
                                              nullptr)
                          << "\n";
                return Vec();
            }

            // left projection.
            if (s == "<." || s == ".<" || s == "lproject" || s == "projectl") {
            }

            return interpret(ast.at(0));
        }

        case ASTTy::Null:
            assert(false && "cannot interpret null ast");
            return Vec();
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
        M[b].normalize();
        printf("%s:", vocab + b * max_w);
        printf(" lensq: %f  ", M[b].lensq());
        for (int i = 0; i < 10; i++) {
            printf("%f ", M[b].ix(i));
        }
        printf("\n");
        // for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1,
        // f); len = 0; for (a = 0; a < size; a++) len += M[a + b * size] * M[a
        // + b * size]; len = sqrt(len); for (a = 0; a < size; a++) M[a + b *
        // size] /= len;
    }

    fclose(f);

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

        Vec v = interpret(ast);
        printvec(v, "vector: ", nullptr);
        cosine(v);
        v.freemem();
    }
}
