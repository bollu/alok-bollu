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
#define N 40
#define max_w 50
#define max(x, y) ((x) > (y) ? (x) : (y))


// and(x, y) = xy
// or(x, y) = x + y - xy
// not(x, y) = 1 - x
// x y = 0
// y = 0/x


// and(x, y) = min x y
// or(x, y) = max x y
// not(x, y) = 1 - x
// and x (not x) = min x (1 - x) = 0.5


using namespace std;
using Vec = float*;

std::map<std::string, Vec> word2vec;

Vec *M;
char *vocab;
long long words, size;

float mk01(float r) {
    return powf(2, r);
}

void normalizeVec(Vec v) {
    float totalsize = 0;
    for(int i = 0; i < size; ++i) totalsize += v[i];
    for(int i = 0; i < size; ++i) v[i] /= totalsize;
}


void plotHistogram(const char *name, float *vals, int n, int nbuckets) {
    // number of values in each bucket.
    int buckets[nbuckets];
    for(int i = 0; i < nbuckets; ++i) buckets[i] = 0;

    float vmax = vals[0];
    float vmin = vals[0];
    for(int i = 0; i < n; ++i) vmax = vals[i] > vmax ? vals[i] : vmax;
    for(int i = 0; i < n; ++i) vmin = vals[i] < vmin ? vals[i] : vmin;

    float multiple = (vmax - vmin) / nbuckets;

    for(int i = 0; i < n; ++i) {
        int b = floor((vals[i] - vmin) / multiple);
        b = b >= nbuckets ? (nbuckets -1): (b < 0 ? 0 : b);
        buckets[b]++;
    }
    
    int total = 0;
    for(int i = 0; i < nbuckets; ++i) total += buckets[i];

    printf("%s: |", name);
    for(int i = 0; i < nbuckets; ++i) {
        printf(" %f ", ((buckets[i] / (float)total)) * 100.0);
    }
    printf("|");

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

float entropylog(float x) {
    if (x < 1e-4) {
        return 0;
    }
    return log(x);
}

float entropy(Vec v) {

    float totalsize = 0;
    for(int i = 0; i < size; ++i) totalsize += v[i];
    for(int i = 0; i < size; ++i) v[i] /= totalsize;

    float H = 0;
    for(int i = 0; i < size; ++i) 
        H += -v[i] * entropylog(v[i]) - (1 - v[i]) * entropylog(1 - v[i]);
    return H;
}



Vec interpret(AST ast) {

    const float STRONG_THRESHOLD = 0.5 / size;
    std::cout << "interpreting: ";
    ast.print();
    std::cout << std::endl;

    switch (ast.ty()) {
        case ASTTy::AtomString: {
            auto it = word2vec.find(ast.s());
            if (it == word2vec.end()) {
                cout << "unable to find word: |" << ast.s() << "|\n";
                goto INTERPRET_ERROR;
            } else {
                cout << ast.s() << " : ";
                Vec out = new float[size];
                for(int i = 0; i < size; ++i) out[i] = it->second[i];

                for(int i = 0; i < std::min<int>(3, size); i++) {
                    cout << setprecision(1) << out[i] << " ";
                }
                cout << "\n";
                return out;
            }
            assert(false && "unreachable");
        }

        case ASTTy::List: {
          if (ast.size() == 0) goto INTERPRET_ERROR;

          if (ast.at(0).ty() != ASTTy::AtomString) {
              cout << "head of AST must be command";
              cout << "\n\t"; ast.print();
              goto INTERPRET_ERROR;
          }

          const std::string command = ast.at(0).s();

          if (command == "and") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(int i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  for(int j = 0; j < size; ++j) {
                      out[j] *= w[j];
                  }
              }

              return out;
          } 
          else if (command == "or") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(int i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  for(int j = 0; j < size; ++j) {
                      out[j] = out[j] +  w[j] - out[j] * w[j];
                  }
              }

              return out;
          } 
          else if (command == "min") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(int i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  for(int j = 0; j < size; ++j) {
                      out[j] = min(out[j], w[j]);
                  }
              }

              return out;
          }
          else if (command == "max") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(int i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  for(int j = 0; j < size; ++j) {
                      out[j] = max(out[j], w[j]);
                  }
              }

              return out;
          }
          // https://en.wikipedia.org/wiki/Fuzzy_set#Fuzzy_set_operations
          else if (command == "difference" || command == "diff") {
              if (ast.size() != 3) {
                  cout << "usage: difference <w1> <w2>\n";
                  goto INTERPRET_ERROR;
              }

              Vec l = interpret(ast.at(1));
              Vec r = interpret(ast.at(2));

              Vec out = new float[size];
              for(int i = 0; i < size; ++i) {
                  out[i] = l[i] - min(l[i], r[i]);
              }
              normalizeVec(out);
              return out;
          }

          else if (command == "analogy") {
              if (ast.size() != 4) {
                  cout << "usage: analogy <w1> <w2> <w3?\n";
                  goto INTERPRET_ERROR;
              }

              // a : b :: x : ?
              Vec a = interpret(ast.at(1));
              Vec b = interpret(ast.at(2));
              Vec x = interpret(ast.at(3));
              normalizeVec(a);
              normalizeVec(b);
              normalizeVec(x);

              Vec out = new float[size];
              for(int i = 0; i < size; ++i) {
                  // a : b :: x : ?
                  // (A U X) / B
                  float delta = b[i] + x[i] - min(b[i] + x[i], a[i]);
                  out[i] = delta;
              }
              normalizeVec(out);
              return out;
          }

         // https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic
          else if (command == "strongmax") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(int i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  normalizeVec(out);
                  normalizeVec(w);

                  for(int j = 0; j < size; ++j) {
                      out[j] = max<float>(0.0, out[j] + w[j] - 1.0 / size);
                  }
              }

              return out;
          }
         // https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic
          else if (command == "strongmin") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(int i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  normalizeVec(out);
                  normalizeVec(w);

                  for(int j = 0; j < size; ++j) {
                      out[j] = min<float>(0, out[j] + w[j]);
                  }
              }

              return out;
          }
          else if (command == "not") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(int j = 0; j < size; ++j) {
                  out[j] = 1 - out[j];
              }

              return out;


          } else if (command == "entropy") {
              float H = 0;
              Vec out = interpret(ast.at(1));
              H += entropy(out);

              cout << "entropy: " << setprecision(5) <<  H << "\n";
              return nullptr;

          } else {
              cout << "unknown command: " << command;
              cout << "\n\t"; ast.print();
              goto INTERPRET_ERROR;
          }

          assert(false && "unreachable");
        }

        case ASTTy::Null:
        default:
            goto INTERPRET_ERROR;

    }

INTERPRET_ERROR:
    return nullptr;
}

// completions for linenoise
void completion(const char *buf, linenoiseCompletions *lc) {
    int ix = strlen(buf) - 1;
    for(; ix >= 0; ix--) {
        if (buf[ix] == ' ' || buf[ix] == '(' || buf[ix] == ')') {
            break;
        }
    }

    ix++;
    if (ix == strlen(buf)) { return; }

    for (int i = words - 1; i >= 0; i--)  {
        char *w = vocab + i * max_w;
        if (strstr(w, buf + ix) == w) {
            // take buf till ix
            std::string completion(buf, ix);
            completion += std::string(w);

            char *ccompletion = new char[completion.size()+2];
            strcpy(ccompletion, completion.c_str());

            linenoiseAddCompletion(lc, ccompletion);
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
            const float cur =  M[w][i];
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

void printCloseWords(Vec vec) {
    float vecsize = 0;
    for (int a = 0; a < size; a++) vecsize += vec[a];
    for (int a = 0; a < size; a++) vec[a] /= vecsize;

    float vals[words];
    float bestd[words];
    char bestw[N][max_size];

    for (int a = 0; a < N; a++) bestd[a] = 0;
    for (int a = 0; a < N; a++) bestw[a][0] = 0;
    for (int c = 0; c < words; c++) {
      float intersectsize = 0;
      for (int a = 0; a < size; a++) {
          intersectsize += vec[a] * M[c][a];
      }
      const float dist  = intersectsize / vecsize;
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
    printf("\n");
}

void printAscByEntropy() {
    printf("Words sorted by entropy (lowest):\n");
    float vals[words];
    float bestd[words];
    char bestw[N][max_size];

    float minentropy = -1;
    for (int a = 0; a < N; a++) bestd[a] = 100;
    for (int a = 0; a < N; a++) bestw[a][0] = 0;
    for (int c = 0; c < words; c++) {
      const float dist = entropy(M[c]);

      for (int a = 0; a < N; a++) {
        if (dist < bestd[a]) {
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
    plotHistogram("entropies", vals, words, 10);
    printf("\n");
}


void printDescByEntropy() {
    printf("Words sorted by entropy (highest):\n");
    float vals[words];
    float bestd[words];
    char bestw[N][max_size];

    float minentropy = -1;
    for (int a = 0; a < N; a++) bestd[a] = -100;
    for (int a = 0; a < N; a++) bestw[a][0] = 0;
    for (int c = 0; c < words; c++) {
      const float dist = entropy(M[c]);

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
    plotHistogram("entropies", vals, words, 10);
    printf("\n");
}

void printDistributionOfProbs() {
    static const int NPROBS = 100;
    int probs[NPROBS];
    for(int i = 0; i < NPROBS; ++i) probs[i] = 0;

    // focus on the probability range from 0 to 0.2
    // 0 -> 0
    // 0.2 -> NPROBS
    for(int i = 0; i < words; ++i) {
        for(int j = 0; j < size; ++j) {
            const float f = M[i][j];
            probs[int(floor(f * (NPROBS+1) / 0.2))]++;
        }
    }

    int total = 0;
    for(int i = 0; i < NPROBS; ++i) total += probs[i];

    cout << "probability distribution across all vectors:\n";
    for(int i = 0; i < NPROBS; ++i) {
        cout << setprecision(2) << float(i) * 0.2 / (NPROBS) << " : " << probs[i]  << "\n";
    }
    cout << "----\n";
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

    M = (Vec *)malloc((long long)words * sizeof(Vec));

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
        M[b] = new float[size];
        printf("%s\n", vocab + b * max_w);

        float setsize = 0;
        for(int i = 0; i < size; ++i) {
            float fl;
            fread(&M[b][i], sizeof(float), 1, f);
            M[b][i] = mk01(M[b][i]);
            setsize += M[b][i];
        }
        
        for(int i = 0; i < size; ++i) {
            M[b][i] /= setsize;
        }

        word2vec[std::string(vocab+b*max_w)] = M[b];


    }


    fclose(f);

    printAscByEntropy();
    printDescByEntropy();
    printDistributionOfProbs();


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

        const Vec v = interpret(ast);
        if (!v) continue;

        printCloseWords(v);
    }
}
