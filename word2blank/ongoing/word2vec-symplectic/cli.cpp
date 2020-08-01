#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include "linenoise.h"
#include <algorithm>
#include <assert.h>
#include <map>
#include <iomanip>
#include<algorithm>


#define max_size 2000
#define N 20
#define max_w 50
#define max(x, y) ((x) > (y) ? (x) : (y))



using namespace std;
struct Vec {
  int size;
  float *v;
  Vec(int size, float *v) : size(size), v(v) {};
  Vec() : size(0), v(nullptr) {};

  Vec(const Vec &other) : size(other.size) {
    v = new float[size];
    for(int i = 0; i < size; ++i) v[i] = other.v[i];
  }
  static Vec alloc(int size) {
    return Vec(size, new float[size]);
  }

  float &operator [](size_t ix) { return v[ix]; }

  // true if size != 0
  operator bool () const { if (size == 0) { return false; } return true; }

  void free() { size = 0; delete[] v; v = nullptr; }
};

std::map<std::string, Vec> word2vec;

Vec *M;
char *g_vocab;
long long g_words, g_size;



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
            case ASTTy::List: {
                std::cout << '('; 
                for(int i = 0; i < list_.size(); ++i) {
                    list_[i].print();
                    if (i < list_.size() - 1) std::cout << ' ';
                }
                std::cout << ')';
                return;
            }
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

    AST operator [](int i) { return at(i); }

    int size() {
        assert(ty_ == ASTTy::List);
        return list_.size();
    }
};

std::tuple<AST, char *> parse_(char *str) {
    assert(str != nullptr);

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


AST parse(char *str) { return std::get<0>(parse_(str)); }

// ===forward declare all commands here===
// ===forward declare all commands here===
// ===forward declare all commands here===
// ===forward declare all commands here===
// ===forward declare all commands here===
// ===forward declare all commands here===
// ===forward declare all commands here===
Vec along(AST ast);

Vec interpret(AST ast) {
    std::cout << "interpreting: ";
    ast.print();
    std::cout << std::endl;

    switch (ast.ty()) {
        case ASTTy::AtomString: {
            auto it = word2vec.find(ast.s());
            if (it == word2vec.end()) {
                cout << "unable to find word: |" << ast.s() << "|\n";
                goto INTERPRET_ERROR;
            }
            cout << ast.s() << " : ";
            Vec out = Vec::alloc(g_size);
            for(int i = 0; i < g_size; ++i) out[i] = it->second[i];
            for(int i = 0; i < std::min<int>(3, g_size); i++) {
                cout << setprecision(1) << out[i] << " ";
            }
            cout << "\n";
            return out;
            goto INTERPRET_ERROR;
        }

        case ASTTy::List: {
          if (ast.size() == 0) goto INTERPRET_ERROR;

          if (ast.at(0).ty() != ASTTy::AtomString) {
              cout << "head of AST must be command";
              cout << "\n\t"; ast.print();
              goto INTERPRET_ERROR;
          }

          const std::string command = ast.at(0).s();

          if(command == "along") { return along(ast); }
          goto INTERPRET_ERROR;
      }

        case ASTTy::Null:
        default:
            goto INTERPRET_ERROR;

    }

INTERPRET_ERROR:
    return Vec();
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
    fscanf(f, "%lld", &g_words);
    fscanf(f, "%lld", &g_size);
    g_vocab = (char *)malloc((long long)g_words * max_w * sizeof(char));

    M = (Vec *)malloc((long long)g_words * sizeof(Vec));

    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)g_words * g_size * sizeof(float) / 1048576, g_words, g_size);
        return -1;
    }
    for (int b = 0; b < g_words; b++) {
        int a = 0;
        while (1) {
            g_vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (g_vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (g_vocab[b * max_w + a] != '\n')) a++;
        }
        g_vocab[b * max_w + a] = 0;
        M[b] = Vec::alloc(g_size);
        printf("%s\n", g_vocab + b * max_w);

        for(int i = 0; i < g_size; ++i) {
            fread(&M[b][i], sizeof(float), 1, f);
        }
        
        word2vec[std::string(g_vocab+b*max_w)] = M[b];
    }


    fclose(f);

    linenoiseHistorySetMaxLen(10000);
    while (1) {
        char *s = linenoise(">");
        linenoiseHistoryAdd(s);
        AST ast = parse(s);
        linenoiseFree(s);
        std::cout << "ast: ";
        ast.print();
        std::cout << "\n\n";

        if (ast.ty() == ASTTy::Null) continue;
        Vec v = interpret(ast);
        v.free();
    }
}


// ===command implementation===
// ===command implementation===
// ===command implementation===
// ===command implementation===
// ===command implementation===
// ===command implementation===
// ===command implementation===

float getlen(float *fs, int size) {
    float len = 0;
    for(int i = 0; i < size; ++i) len += fs[i] * fs[i];
    len = sqrt(len);
    return len;
}
void normalize(float *fs, int size) {
    float len = getlen(fs, size);
    for(int i = 0; i < size; ++i) fs[i] /= len;
}

void printCloseWords(Vec v) {
    float vals[g_words];
    float bestd[g_words];
    char bestw[N][max_size];

    for (int a = 0; a < N; a++) bestd[a] = 0;
    for (int a = 0; a < N; a++) bestw[a][0] = 0;
    for (int c = 0; c < g_words; c++) {
        float dist1 = 0;
        for (int a = 0; a < g_size/2; a++) {
            dist1 += v[a] * M[c][a];
        }
        dist1 /= getlen(M[c].v, g_size/2);
        dist1 /= getlen(v.v, g_size/2);

        float dist2 = 0;
        for (int a = g_size/2; a < g_size; a++) {
            dist2 += v[a] * M[c][a];
        }
        dist2 /= getlen(v.v + g_size/2, g_size/2);
        dist2 /= getlen(M[c].v + g_size/2, g_size/2);

        const float dist = dist1 + dist2;

        for (int a = 0; a < N; a++) {
            if (dist > bestd[a]) {
                for (int d = N - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], &g_vocab[c * max_w]);
                break;
            }
        }
    }
    for (int a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    // plotHistogram("distances", vals, g_words, 10);
    printf("\n");
}

void printCloseWordsHalf(Vec v, bool h) {
    float vals[g_words];
    float bestd[g_words];
    char bestw[N][max_size];
    const int OFFSET = h*g_size/2;

    for (int a = 0; a < N; a++) bestd[a] = 0;
    for (int a = 0; a < N; a++) bestw[a][0] = 0;
    for (int c = 0; c < g_words; c++) {
      float dist = 0;
      for (int a = OFFSET; a < OFFSET + g_size/2; a++) {
          dist += v[a] * M[c][a];
      }
      // normalize by length.
      dist /= getlen(M[c].v + OFFSET, g_size/2);
      dist /= getlen(v.v + OFFSET, g_size/2);

      for (int a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (int d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &g_vocab[c * max_w]);
          break;
        }
      }
    }
    for (int a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
    printf("\n");
}

Vec along(AST ast) {
    assert(ast.ty() == ASTTy::List);
    if(ast.size() != 3) {
        cout << "usage: (along <w1> <w2>)\n"; return Vec();
    }

    Vec v = interpret(ast[1]);
    Vec w = interpret(ast[2]);
    if (!v || !w) { return Vec(); }
    assert(v.size == g_size);
    assert(w.size == g_size);

    for(int h = 0; h <= 1; ++h) {
        const int hcomp = 1 - h;
        for(int dt = -1000; dt < 1000; ++dt) {
            Vec x(v);
            for(int i = 0; i < (g_size/2); ++i) {
                x[h*(g_size/2) + i] += w[hcomp*(g_size/2) + i] * dt * 0.01;
            }
            printf("***dt = %d | h = %d *****\n", dt, h);
            printCloseWordsHalf(x, /*top half=*/h);
            x.free();
        }
    }

    /*
    for(int dt = -1000; dt < 1000; ++dt) {
        Vec x(v);
        for(int i = 0; i < g_size; ++i) {
            x[i] += w[i] * dt * 0.001;
        }
        printf("***dt = %d | h = BOTH *****\n", dt);
        printCloseWords(x);
        x.free();
    }
    */

    v.free();
    w.free();
    return Vec();
}
