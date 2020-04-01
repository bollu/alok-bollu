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
#include <set>
#include<algorithm>


static const long long MINFREQ = 0;
static const long long FUNCTION_WORD_FREQ_CUTOFF = 40;
#define N 40
#define max_w 500
#define max(x, y) ((x) > (y) ? (x) : (y))


double *vals;// [words];
double *bestd;//[words];
char *bestw;// [N][max_w];


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
using Vec = double*;

std::map<std::string, Vec> word2vec;

// requires the file freq-text8.txt
map<string, long long> word2freq;

// function words, detected automatically
set<string> functionwords;

// map from entropy to a word index
map<double, long long>  entropy2w; 
map<long long, double> word2entropy;

Vec *M;
Vec *Ml;
Vec *Mloneminus;
char *vocab;
long long words, size;

double sigmoid(double r) {
    return powf(2, r) / (1 + powf(2, r));
}



void normalizeVec(Vec v) {
    assert(false && "one should not need this");

    double totalsize = 0;
    for(long long i = 0; i < size; ++i) totalsize += v[i];
    for(long long i = 0; i < size; ++i) v[i] /= totalsize;
}


void plotHistogram(const char *name, double *vals, long long n, long long nbuckets) {
    // number of values in each bucket.
    long long buckets[nbuckets];
    for(long long i = 0; i < nbuckets; ++i) buckets[i] = 0;

    double vmax = vals[0];
    double vmin = vals[0];
    for(long long i = 0; i < n; ++i) vmax = vals[i] > vmax ? vals[i] : vmax;
    for(long long i = 0; i < n; ++i) vmin = vals[i] < vmin ? vals[i] : vmin;

    double multiple = (vmax - vmin) / nbuckets;

    for(long long i = 0; i < n; ++i) {
        long long b = floor((vals[i] - vmin) / multiple);
        b = b >= nbuckets ? (nbuckets -1): (b < 0 ? 0 : b);
        buckets[b]++;
    }
    
    long long total = 0;
    for(long long i = 0; i < nbuckets; ++i) total += buckets[i];

    printf("%s: |", name);
    for(long long i = 0; i < nbuckets; ++i) {
        printf(" %f ", ((buckets[i] / (double)total)) * 100.0);
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
                for(long long i = 0; i < (long long)list_.size(); ++i) {
                    list_[i].print();
                    if (i < (long long)list_.size() - 1) std::cout << ' ';
                }
                std::cout << ')';
                return;
            }
            case ASTTy::Null:
                std::cout << "null";
        }
    }

    ASTTy ty() { return ty_; };
    std::string s() { assert(ty_ == ASTTy::AtomString); return s_; };
    double f() { assert(ty_ == ASTTy::AtomString); return atof(s_.c_str()); }
    long long i() { assert(ty_ == ASTTy::AtomString); return atoi(s_.c_str()); }

    AST at(long long i) {
        assert(ty_ == ASTTy::List);
        return list_[i];
    }

    long long size() {
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
        long long offset = 0;
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

double entropylog(double x) {
    if (x < 1e-8) return 0;
    return log(x);
}

double entropy(Vec v) {

    double H = 0;
    for(long long i = 0; i < size; ++i) 
        H += -v[i] * entropylog(v[i]) - (1 - v[i]) * entropylog(1 - v[i]);
    return H;
}

// crossentropy(P, q) = H(p) + KL(p, q)
// https://juaa-journal.springeropen.com/track/pdf/10.1186/s40467-015-0029-5
double crossentropy(Vec v, Vec w) {
    double H = 0;
    for(long long i = 0; i < size; ++i)  {
        if (w[i] < 1e-300L) w[i] = 1e-300L;
        H += v[i] * (entropylog(v[i]) - entropylog(w[i])) + 
            (1 - v[i]) * (entropylog((1 - v[i])) - entropylog((1-w[i])));
    }
    return H;
}

double kl(Vec v, Vec w) {
    double H = 0;
    for(long long i = 0; i < size; ++i)  {
        if (w[i] < 1e-300L) w[i] = 1e-300L;
        H += -v[i] * entropylog(w[i]) - (1 - v[i]) *  entropylog((1-w[i]));
    }
    return H;
}

// completions for linenoise
void completion(const char *buf, linenoiseCompletions *lc) {
    long long ix = strlen(buf) - 1;
    for(; ix >= 0; ix--) {
        if (buf[ix] == ' ' || buf[ix] == '(' || buf[ix] == ')') {
            break;
        }
    }

    ix++;
    if (ix == (long long)strlen(buf)) { return; }

    for (long long i = words - 1; i >= 0; i--)  {
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

    for (long long i = 0; i < size; i++) {
        f[i] = 0;
        fnorm[i] = 0;
    }
    for (long long w = 0; w < words; w++) {
        for (long long i = 0; i < size; i++) {
            const double cur =  M[w][i];
            f[i] += fabs(cur);
        }
    }

    double total = 0;
    for (long long i = 0; i < size; ++i) total += f[i];

    // normalize
    for (long long i = 0; i < size; ++i) fnorm[i] = (f[i] * 100.0) / total;

    printf("dimension weights as percentage [0..n]:\n");
    for (long long i = 0; i < size; ++i) printf("%lld: %5.8f\n", i, fnorm[i]);
    printf("\n");
}

void printClosestWordsSetOverlap(Vec vec, Vec *M) {
    printf("DISTANCE ASYMMETRIC\n");
    double vecsize = 0;
    for (long long a = 0; a < size; a++) vecsize += vec[a];

    for (long long a = 0; a < N; a++) bestd[a] = 0;
    for (long long a = 0; a < N; a++) bestw[a*max_w] = '\0';
    printf("\n");
    for (long long c = 0; c < words; c++) {
      printf("\r%4lld / %4lld: %4.2f%%", c, words, 100.0 * ((float)c/words));
      double intersectsize = 0;
      for (long long a = 0; a < size; a++) {
          intersectsize += vec[a] * M[c][a];
      }
      const double dist  = intersectsize / vecsize;
      vals[c] = dist;

      for (long long a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (long long d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw+d*max_w, bestw+(d - 1)*max_w);
          }

          bestd[a] = dist;
          strcpy(bestw + a*max_w, &vocab[c * max_w]);
          break;
        }
      }
    }
    for (long long a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw+a*max_w, bestd[a]);
    plotHistogram("distances", vals, words, 10);
    printf("\n");
}


void printClosestWordsSetOverlapSymmetric(Vec vec, Vec *M) {
    printf("DISTANCE SYMMETRIC\n");

    for (long long a = 0; a < N; a++) bestd[a] = 0;
    for (long long a = 0; a < N; a++) bestw[a*max_w]= '\0';
    printf("\n");
    for (long long c = 0; c < words; c++) {
      printf("\r%4lld / %4lld: %4.2f%%", c, words, 100.0 * ((float)c/words));
      double intersectsize = 0;
      double unionsize = 0;
      for (long long a = 0; a < size; a++) {
          intersectsize += vec[a] * M[c][a];
          unionsize += vec[a] + M[c][a] - vec[a] *M[c][a];
      }
      assert(intersectsize >= 0);
      assert(unionsize >= 0);
      const double dist  = intersectsize / unionsize;
      vals[c] = dist;

      for (long long a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (long long d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw+d*max_w, bestw+(d - 1)*max_w);
          }
          bestd[a] = dist;
          strcpy(bestw + a*max_w, &vocab[c * max_w]);
          break;
        }
      }
    }
    for (long long a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw+a*max_w, bestd[a]);
    plotHistogram("distances", vals, words, 10);
    printf("\n");
}


void printClosestWordsCrossEntropy(Vec vec, Vec *M) {
    printf("CROSS ENTROPY: EXTRA BITS FOR word on FOCUS\n");
    double *vals = new double [words];
    double *bestd = new double[words];

    for (long long a = 0; a < N; a++) bestd[a] = 1000;
    for (long long a = 0; a < N; a++) bestw[a*max_w] = '\0';
    printf("\n");
    for (long long c = 0; c < words; c++) {
      printf("\r%4lld / %4lld: %4.2f%%", c, words, 100.0 * ((float)c/words));
      const double dist = crossentropy(M[c], vec);
      vals[c] = dist;
      string w = string(vocab +c*max_w);

      for (long long a = 0; a < N; a++) {
        if (dist < bestd[a]) {
          for (long long d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw+d*max_w, bestw+(d - 1)*max_w);
          }
          bestd[a] = dist;
          strcpy(bestw + a*max_w, &vocab[c * max_w]);
          break;
        }
      }
    }
    for (long long a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw+a*max_w, bestd[a]);
    plotHistogram("distances", vals, words, 10);
    printf("\n");

    delete []vals;
    delete []bestd;
}

void printClosestWordsCrossEntropy2(Vec vec, Vec *M) {
    printf("CROSS ENTROPY: EXTRA BITS FOR FOCUS on word\n");
    double *vals = new double [words];
    double *bestd = new double[words];

    for (long long a = 0; a < N; a++) bestd[a] = 100;
    for (long long a = 0; a < N; a++) bestw[a*max_w] = '\0';
    printf("\n");
    for (long long c = 0; c < words; c++) {
      printf("\r%4lld / %4lld: %4.2f%%", c, words, 100.0 * ((float)c/words));
      const double dist = crossentropy(vec, M[c]);
      vals[c] = dist;
      string w = string(vocab +c*max_w);

      for (long long a = 0; a < N; a++) {
        if (dist < bestd[a]) { // && word2freq[w] > MINFREQ) {
          for (long long d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw+d*max_w, bestw+(d - 1)*max_w);
          }
          bestd[a] = dist;
          strcpy(bestw + a*max_w, &vocab[c * max_w]);
          break;
        }
      }
    }
    for (long long a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw+a*max_w, bestd[a]);
    plotHistogram("distances", vals, words, 10);
    printf("\n");

    delete []vals;
    delete []bestd;
}

void printClosestWordsCrossEntropySym(Vec vec, Vec *M) {
    printf("CROSS ENTROPY: BOTH\n");
    double *vals = new double [words];
    double *bestd = new double[words];
    double *dist1 = new double[words];
    double *dist2 = new double[words];
    double maxdist1 = -10, maxdist2 = -10;

    for (long long a = 0; a < N; a++) bestd[a] = 1000;
    for (long long a = 0; a < N; a++) bestw[a*max_w] = '\0';
    for (long long c = 0; c < words; c++) {
        dist1[c] = crossentropy(vec, M[c]);
        if (dist1[c] > maxdist1) maxdist1 = dist1[c];
        dist2[c] =  crossentropy(M[c], vec);
        if (dist2[c] > maxdist2) maxdist2 = dist2[c];
      // vals[c] = dist;
    }

    // make sure its's not negative...
    // if (maxdist1 < 0) maxdist1 = 1;
    // if (maxdist2 < 0) maxdist2 = 1;
    
    maxdist1 = maxdist2 = 1;

    for (long long c = 0; c < words; c++) {
      // const double dist = (dist1[c] / maxdist1) + (dist2[c] / maxdist2);
      const double dist = dist1[c] + dist2[c];
      vals[c] = dist;
      string w = string(vocab +c*max_w);
      for (long long a = 0; a < N; a++) {
        if (dist < bestd[a] && word2freq[w] > MINFREQ) {
          for (long long d = N - 1; d > a; d--) {
            bestd[d] = bestd[d-1];
            strcpy(bestw+d*max_w, bestw+(d - 1)*max_w);
          }
          bestd[a] = dist;
          strcpy(bestw + a*max_w, &vocab[c * max_w]);
          break;
        }
      }
    }
    for (long long a = 0; a < N; a++) {
        printf("%50s\t\t%f\n", bestw+a*max_w, bestd[a]);

        if (a < N - 1 && fabs(bestd[a+1] - bestd[a]) > 1e-3) {
            printf("-------------------------------------------------------\n");
        }
    }
    plotHistogram("distances", vals, words, 10);
    printf("\n");

    delete []vals;
    delete []bestd;
    delete []dist1;
    delete []dist2;
}


void printClosestWordsKL(Vec vec, Vec *M) {
    printf("KL divergence (always an answer, may not be what you're looking for)\n");
    double *vals = new double [words];
    double *bestd = new double[words];

    for (long long a = 0; a < N; a++) bestd[a] = 999999;
    for (long long a = 0; a < N; a++) bestw[a*max_w] = '\0';
    printf("\n");
    for (long long c = 0; c < words; c++) {
        printf("\r%4lld / %4lld: %4.2f%%", c, words, 100.0 * ((float)c/words));
      // const double dist = crossentropy(vec, M[c]);
      string w = string(vocab +c*max_w);
      const double dist = kl(M[c], vec);
      vals[c] = dist;

      for (long long a = 0; a < N; a++) {
        if (dist < bestd[a]) { // && word2freq[w] > MINFREQ) {
          for (long long d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw+d*max_w, bestw+(d - 1)*max_w);
          }
          bestd[a] = dist;
          strcpy(bestw+a*max_w, &vocab[c * max_w]);
          break;
        }
      }
    }
    for (long long a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw+a*max_w, bestd[a]);
    plotHistogram("distances", vals, words, 10);
    printf("\n");

    delete []vals;
    delete []bestd;
}

void computeEntropies(Vec *M) {
    printf ("computing entropies...\n");
    for (long long c = 0; c < words; c++) {
      const double H = entropy(M[c]);
      word2entropy[c] = H;
      entropy2w[H] = c;
      printf("\r%4lld / %4lld: %4.2f%%", c, words, 100.0 * ((float)c/words));
    }
}
void printAscByEntropy(Vec *M, long long freq_cutoff) {
    printf("Words sorted by entropy (lowest):\n");

    for (long long a = 0; a < N; a++) bestd[a] = 1000;
    for (long long a = 0; a < N; a++) bestw[a*max_w] = '\0';
    for (long long c = 0; c < words; c++) {
      const double dist = entropy(M[c]);
      string w = string(vocab +c*max_w);

      for (long long a = 0; a < N; a++) {
        if (dist < bestd[a] && word2freq[w] > freq_cutoff) {
          for (long long d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw+d*max_w, bestw+(d - 1)*max_w);
          }
          bestd[a] = dist;
          strcpy(bestw + a*max_w, &vocab[c * max_w]);
          break;
        }
      }
    }
    for (long long a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw+a*max_w, bestd[a]);
    plotHistogram("entropies", vals, words, 10);
    printf("\n");
}


void printDescByEntropy(Vec *M, long long minfreq) {
    printf("Words sorted by entropy (highest)/ function words:\n");

    for (long long a = 0; a < N; a++) bestd[a] = -1000;
    for (long long a = 0; a < N; a++) bestw[a*max_w] = '\0';
    for (long long c = 0; c < words; c++) {
      const double dist = entropy(M[c]);
      string w = string(vocab +c*max_w);

      for (long long a = 0; a < N; a++) {
        if (dist > bestd[a] && word2freq[w] > minfreq) {
          for (long long d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw+d*max_w, bestw+(d - 1)*max_w);
          }
          bestd[a] = dist;
          strcpy(bestw + a*max_w, &vocab[c * max_w]);
          break;
        }
      }
    }
    for (long long a = 0; a < N; a++) {
        printf("%50s\t\t%f\n", bestw + a*max_w, bestd[a]);
        functionwords.insert(bestw + a*max_w);
    }
    plotHistogram("entropies", vals, words, 10);
    printf("\n");
}

// useful for making graphs
void printWordsAtEntropy(Vec *M, double center) {
    printf("Words at entropy (%f)\n", center);

    for (long long a = 0; a < N; a++) bestd[a] = 1000;
    for (long long a = 0; a < N; a++) bestw[a*max_w] = '\0';
    for (long long c = 0; c < words; c++) {
      const double H = entropy(M[c]);
      const double dist = (center -  H) * (center - H);
      string w = string(vocab +c*max_w);

      for (long long a = 0; a < N; a++) {
        if (dist < bestd[a] && word2freq[w] > MINFREQ) {
          for (long long d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw+d*max_w, bestw+(d - 1)*max_w);
          }
          bestd[a] = dist;
          strcpy(bestw + a*max_w, &vocab[c * max_w]);
          break;
        }
      }
    }
    for (long long a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw+a*max_w, bestd[a]);
    plotHistogram("entropies", vals, words, 10);
    printf("\n");
}


void detectPolysemousWords() {
    cout << "polysemous words are:\n";
}

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
            } else {
                cout << ast.s() << " : ";
                Vec out = new double[size];
                for(long long i = 0; i < size; ++i) out[i] = it->second[i];

                for(long long i = 0; i < std::min<int>(3, size); i++) {
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

              for(long long i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  for(long long j = 0; j < size; ++j) {
                      out[j] *= w[j];
                  }
              }

              return out;
          } else if (command == "/" || command == "div") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(long long i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  for(long long j = 0; j < size; ++j) {
                      out[j] = max<double>(min<double>(out[j] / (1e-3 + w[j]), 1 - 1e-3), 1e-3);
                  }
              }

              printf("div: ");
              for(long long j = 0; j < size; ++j) {
                  printf("%4.2f ", out[j]);
              }
              printf("\n");
              return out;

          } else if (command == "or" || command == "+") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(long long i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  for(long long j = 0; j < size; ++j) {
                      /// out[j] = 1 - (1 - out[j]) * (1 - w[j]);
                      out[j] = out[j] + w[j] - out[j] * w[j];
                  }
              }

              return out;
          } 
          else if (command == "min") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(long long i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  for(long long j = 0; j < size; ++j) {
                      out[j] = min(out[j], w[j]);
                  }
              }

              return out;
          }
          else if (command == "max") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(long long i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;

                  for(long long j = 0; j < size; ++j) {
                      out[j] = max(out[j], w[j]);
                  }
              }

              return out;
          }
          // https://en.wikipedia.org/wiki/Fuzzy_set#Fuzzy_set_operations
          else if (command == "difference" || command == "diff" || command == "-") {
              if (ast.size() != 3) {
                  cout << "usage: difference <w1> <w2>\n";
                  goto INTERPRET_ERROR;
              }

              Vec l = interpret(ast.at(1));
              Vec r = interpret(ast.at(2));

              if (!l || !r) goto INTERPRET_ERROR;

              Vec out = new double[size];
              for(long long i = 0; i < size; ++i) {
                  out[i] = l[i] - min(l[i], r[i]); // ORIGINAL
                  // out[i] = max(0, l[i] + r[i] - 2 * l[i] * r[i]); 
              }
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

              if (!a || !b || !x) goto INTERPRET_ERROR;


              Vec out = new double[size];
              for(long long i = 0; i < size; ++i) {
                  // a : b :: x : ?
                  // (A U X) / B
                  double delta = b[i] + x[i] - min(b[i] + x[i], a[i]); // ORIGINAL
                  delta =  max(delta, 0.0);
                  delta = min(delta, 1.0);
                  out[i] = delta;
              }
              return out;
          }

         // https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic
          else if (command == "strongmax") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(long long i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;


                  for(long long j = 0; j < size; ++j) {
                      out[j] = max<double>(0.0, out[j] + w[j] - 1.0 / size);
                  }
              }

              return out;
          }
         // https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic
          else if (command == "strongmin") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(long long i = 2; i < ast.size(); ++i) {
                  Vec w = interpret(ast.at(i));
                  if (!w) goto INTERPRET_ERROR;


                  for(long long j = 0; j < size; ++j) {
                      out[j] = min<double>(0, out[j] + w[j]);
                  }
              }

              return out;
          }
          else if (command == "not") {
              Vec out = interpret(ast.at(1));
              if (!out) goto INTERPRET_ERROR;

              for(long long j = 0; j < size; ++j) {
                  out[j] = min(1.0, max(1 - out[j], 0.0));
              }

              return out;


          } else if (command == "entropy") {
              double H = 0;
              Vec out = interpret(ast.at(1));
              H += entropy(out);

              cout << "entropy: " << setprecision(5) <<  H << "\n";
              return nullptr;

          } else if (command == "crossentropy") {
              if (ast.size() != 3) goto INTERPRET_ERROR;
              Vec a = interpret(ast.at(1));
              Vec b = interpret(ast.at(2));
              if (!a || !b) goto INTERPRET_ERROR;

              printf("\tcross-entropy: %7.5f\n", crossentropy(a, b));
              fflush(stdout);
              goto INTERPRET_ERROR;

          } else if (command == "kl") {
              if (ast.size() != 3) goto INTERPRET_ERROR;
              Vec a = interpret(ast.at(1));
              Vec b = interpret(ast.at(2));
              if (!a || !b) goto INTERPRET_ERROR;

              printf("\tK-L divergence: %7.5f\n", kl(a, b));
              fflush(stdout);
              goto INTERPRET_ERROR;

          }else if (command == "reldist") {
              if (ast.size() != 3) goto INTERPRET_ERROR;

            Vec *Mrel = new Vec[words];
            Vec rel = interpret(ast.at(1));
            Vec v = interpret(ast.at(2));

            if (!rel) goto INTERPRET_ERROR;
            if (!v) goto INTERPRET_ERROR;

            for(long long i = 0; i < words; ++i){
                Mrel[i] = new double[size];
                for(long long j = 0; j < size; ++j) {
                    Mrel[i][j] = M[i][j] * rel[j];
                }
            }

            for(long long i = 0; i < size; ++i){
                v[i] *= rel[i];
            } 


            printAscByEntropy(M, MINFREQ);
            printDescByEntropy(M, MINFREQ);

            printClosestWordsSetOverlap(v, Mrel);
            printClosestWordsSetOverlapSymmetric(v, Mrel);
            printClosestWordsCrossEntropy(v, Mrel);
            printClosestWordsCrossEntropy2(v, Mrel);
            printClosestWordsCrossEntropySym(v, Mrel);
            printClosestWordsKL(v, Mrel);

            return nullptr;

          } else if (command == "indicator") {
            // probe certain dimensions, by creating vectors with "1" along
            // those dimensions and 0 everywhere else
            Vec indicator = new double[size];
            for(long long i = 0; i < size; ++i) indicator[i] = 0;

            for(long long i = 1; i < ast.size(); ++i) {
                if (ast.at(i).ty() != ASTTy::AtomString) {
                    goto INTERPRET_ERROR;
                }
                const long long ix = ast.at(i).i();
                if (ix < 0 || ix >= size) { goto INTERPRET_ERROR; };

                indicator[ix] = (1) / max<double>(1, (ast.size() - 1));
            }

                printf("indicator: ");
                for(long long i = 0; i < size; ++i) {
                    printf("[%lld]%4.2f ", i, indicator[i]);
                }


            return indicator;
       } else if (command == "mul" || command == "*") {
           // (* vector double)
           if(ast.size() != 3) goto INTERPRET_ERROR;
           Vec v = interpret(ast.at(1));
           if (!v) goto INTERPRET_ERROR;
           if (ast.at(2).ty() != ASTTy::AtomString) goto INTERPRET_ERROR; 

           double f = ast.at(2).f();

           for(long long i = 0; i < size; ++i) {
               v[i] *= f;
           }

           return v;
       } else if (command == "pow" || command == "^") {
           // (* vector double)
           if(ast.size() != 3) goto INTERPRET_ERROR;
           Vec v = interpret(ast.at(1));
           if (!v) goto INTERPRET_ERROR;
           if (ast.at(2).ty() != ASTTy::AtomString) goto INTERPRET_ERROR; 

           double f = ast.at(2).f();

           for(long long i = 0; i < size; ++i) {
               v[i] = powf(v[i], f);
           }

           return v;
       } else if (command == "discrete") {
                if (ast.size() < 2) goto INTERPRET_ERROR;
                Vec v = interpret(ast.at(1));
                if (!v) goto INTERPRET_ERROR;

                double fmax = 0;
                for(long long i = 0; i < size; ++i) {
                    fmax  = max(v[i], fmax);
                }

                // can give threshold factor, which values < fraction *fmax
                // are removed. 
                // By default, is 0.5
                double threshold = 0.5;
                if (ast.size() == 3) {
                    if (ast.at(2).ty() != ASTTy::AtomString) goto INTERPRET_ERROR;
                    threshold = ast.at(2).f();
                }

                long long nlive = 0;
                for(long long i = 0; i < size; ++i) {
                    v[i] = v[i] > fmax * threshold ? 1 : 0;
                    nlive += v[i];
                }

                printf("live: " );
                for(long long i = 0; i < size; ++i) {
                    if (v[i] == 1) printf("%lld ", i);
                }
                printf("\n");

                printf("dead: " );
                for(long long i = 0; i < size; ++i) {
                    if (v[i] == 0) printf("%lld ", i);
                }
                printf("\n");

                for(long long i = 0; i < size; ++i) {
                    v[i] /= nlive;
                }

                printf("discrete: ");
                for(long long i = 0; i < size; ++i) {
                    printf("[%lld]%4.2f ", i, v[i]);
                }
                printf("\n");

                return v;
       } else if (command == "writeprobfile") {
           cout << "\nL" << __LINE__ << std::flush;
           if (ast.size() == 1) {
               cout << "\nL" << __LINE__ << std::flush;
               cout << "writing out coefficients of every vector..." << flush;
               FILE *f = fopen("prob.txt", "w");
               for(long long i = 0; i < words; ++i) {
                   for(long long j = 0; j < size; ++j) {
                       fprintf(f, "%f ", M[i][j]);
                   }
               }
               fclose(f);
               cout << "done.\n";
           } else if (ast.size() == 2) {
               if (ast.at(1).ty() != ASTTy::AtomString) { goto INTERPRET_ERROR; }

               string s = ast.at(1).s();
               Vec v = interpret(ast.at(1));

               if (!v) { goto  INTERPRET_ERROR; }

               char filename[512];
               sprintf(filename, "prob-%s.txt", s.c_str());
               printf("writing out coefficients of given vector to: |%s|", filename);
               fflush(stdout);
               FILE *f = fopen(filename, "w");

               // find index of word.
               long long i = 0;
               while (!strcmp(vocab + max_w  *i, s.c_str())) { continue; }

               for(long long j = 0; j < size; ++j) {
                   fprintf(f, "%f ", M[i][j]);
               }
               fclose(f);
               cout << "done.\n";
           } else {
               assert(false);
           }
          goto INTERPRET_ERROR;
       } else if (command == "writeentropyfile") {
           cout << "writing out entropy of every vector..." << flush;
           FILE *f = fopen("entropy.txt", "w");
           for(long long i = 0; i < words; ++i) {
                   fprintf(f, "%f ", entropy(M[i]));
           }
           fclose(f);
           cout << "done.\n";
          goto INTERPRET_ERROR;
       } else if (command == "entails") {
           // check if the first vector entails the other one
           if (ast.size() != 3) goto INTERPRET_ERROR;
           Vec l = interpret(ast.at(1));
           Vec r = interpret(ast.at(2));
           Vec orv = new double[size];
           for(long long i = 0; i < size; ++i) {
               orv[i] = l[i] + r[i] - l[i] * r[i];
           }

           // find closest vector to orv. if it's lv, then lv => rv. If not,
           // then it doesn't

        
           

       } else if (command == "detectpolysemous") {
           detectPolysemousWords();
           goto INTERPRET_ERROR;
       } else  {
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

int main(int argc, char **argv) {
    char file_name[512];
    FILE *f;

    if (argc != 2) {
        printf("usage: %s <path-to-model.bin>\n", argv[0]);
        return 0;
    }

    strcpy(file_name, argv[1]);
    printf("opening file:|%s|\n", file_name);
    fflush(stdout);

    f = fopen(file_name, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);

    printf("words: %lld | size: %lld\n", words, size);
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    if (!vocab) {
        printf("failed to allocate: %4.2fGB\n",  
                ((long long)words * max_w * sizeof(char)) / 1024.0 / 1024.0 / 1024.0);
    }
    assert(vocab);

    M = (Vec *)malloc((long long)words * sizeof(Vec));
    Ml = (Vec *)malloc((long long)words * sizeof(Vec));
    Mloneminus = (Vec *)malloc((long long)words * sizeof(Vec));

    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)words * size * sizeof(double) / 1048576, words, size);
        return -1;
    }

    assert(M);
    assert(Ml);
    assert(Mloneminus);

    vals = (double*)malloc((long long)words * sizeof(double)); assert(vals);
    bestd = (double*)malloc((long long)words * sizeof(double)); assert(bestd);
    bestw = (char*)malloc((long long)N * max_w * sizeof(char)); assert(bestw);

    for (long long b = 0; b < words; b++) {
        long long a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        vocab[b * max_w + a] = 0;
        M[b] = new double[size];
        printf("\r%4lld / %4lld: %4.2f%%", b, words, 100.0 * ((float)b/words));

        for(int i = 0; i < size; ++i) {
            float fl;
            fread(&fl, sizeof(float), 1, f);
            M[b][i] = fl;
            M[b][i] = powf(2, M[b][i]);
        }
        word2vec[std::string(vocab+b*max_w)] = M[b];
    }


    // printf("\nnormalizing each word vector:\n");
    // for(long long a = 0; a < size; ++a) {
    //     double total = 0;
    //     for(long long b = 0; b < words; b++)  {
    //         total += M[b][a];
    //     }

    //     for(long long b = 0; b < words; b++)  {
    //         M[b][a] /= total;
    //         M[b][a] = max<double>(min<double>(1.0, M[b][a]), 0.0);
    //     }
    //     printf("\r%4lld / %4lld: %4.2f%%", a, size, 100.0 * ((float)a/size));
    // }

    // normalize features per vector
    printf("\nnormalizing along features:\n");
    for(long long b = 0; b < words; ++b) {
        double total = 0;
        for(long long a = 0; a < size; a++)  {
            total += M[b][a];
        }

        for(long long a = 0; a < size; a++)  {
            M[b][a] /= total;
            M[b][a] = max<double>(min<double>(1.0, M[b][a]), 0.0);
        }
        printf("\r%4lld / %4lld: %4.2f%%", b, words, 100.0 * ((float)b/words));
    }



    printf("\ncalculating logarithms...\n");
    for(long long b = 0; b < words; ++b) {
        Ml[b] = new double[size];
        Mloneminus[b] = new double[size];
        for(long long a = 0; a < size; a++) {
            Ml[b][a] = entropylog(M[b][a]);
            Mloneminus[b][a] = entropylog(1.0 - M[b][a]);
        }
        printf("\r%4lld / %4lld: %4.2f%%", b, words, 100.0 * ((float)b/words));
    }

    f = fopen("freq-text8.txt", "r");
    assert (f && "unable to find freq-text8.txt");
    while(!feof(f)) {
        char line[1000];
        fscanf(f, "%s", line);
        char word[1000];
        for(long long j = 0; j < 1000; ++j) word[j] = 0;
        long long i = 0;
        for(i = 0; line[i] != '|'; ++i) {
            word[i] = line[i];
        }
        i++;
        char freqstr[1000];
        strcpy(freqstr, line  + i);
        word2freq[word] = atoi(freqstr);
    }



    computeEntropies(M);
    {
        cout << "\ndescending entropy, dropping words with (freq <" << FUNCTION_WORD_FREQ_CUTOFF <<  " ):\n";
        long long i = 0;
        for(auto it = entropy2w.rbegin(); i < 200 && it != entropy2w.rend(); ++i, ++it) {
            const string w(vocab + max_w *it->second);
            if (word2freq[w] < FUNCTION_WORD_FREQ_CUTOFF) continue;
            printf("%30s\t%f\n", w.c_str(), it->first);
        }
    }


    //printAscByEntropy(M, FUNCTION_WORD_FREQ_CUTOFF);
    //printDescByEntropy(M, FUNCTION_WORD_FREQ_CUTOFF);
    //printWordsAtEntropy(M, 6.26);
    //
    //detectPolysemousWords();

    printf("\ndone processing...\n\n");
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

        printClosestWordsSetOverlap(v, M);
        // printClosestWordsSetOverlapSymmetric(v, M);
        printClosestWordsCrossEntropy(v, M);
        printClosestWordsCrossEntropy2(v, M);
        // printClosestWordsCrossEntropySym(v, M);
        printClosestWordsKL(v, M);
    }
}
