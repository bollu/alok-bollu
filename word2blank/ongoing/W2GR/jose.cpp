#include <stdio.h>
#include <iostream>
#include <libgen.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <assert.h>
#include <stddef.h>
#include <unistd.h>
#include <armadillo>
#include "grad.h"

using namespace std;

#define MAX_STRING 100
#define ACOS_TABLE_SIZE 5000
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int corpus_max_size = 40000000;  // Maximum 40M documents in the corpus

// typedef float real;

struct vocab_word {
    long long cn;
    char *word;
};

char train_file[MAX_STRING], load_emb_file[MAX_STRING];
char word_emb[MAX_STRING], context_emb[MAX_STRING], doc_output[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 20, min_reduce = 1, P = 1, L = 1;
int *vocab_hash;
long long *doc_sizes;
long long vocab_max_size = 1000, vocab_size = 0, corpus_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 10, file_size = 0;
int negative = 2;
const int table_size = 1e8;
int *word_table;
double alpha = 0.04, starting_alpha, sample = 1e-3, margin = 0.10;
double *syn0, *syn1neg, *syn1doc;
arma::cube c_syn0, c_syn1neg, c_syn1doc;
clock_t start;


void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  word_table = (int *) malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    word_table[a] = i;
    if (a / (double) table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *) "</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Locate line number of current file pointer
int FindLine(FILE *fin) {
  long long pos = ftell(fin);
  long long lo = 0, hi = corpus_size - 1;
  while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (doc_sizes[mid] > pos) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) { // assert all sortings will be the same (since c++ qsort is not stable..)
  if (((struct vocab_word *) b)->cn == ((struct vocab_word *) a)->cn) {
    return strcmp(((struct vocab_word *) b)->word, ((struct vocab_word *) a)->word);
  }
  return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

int IntCompare(const void * a, const void * b) 
{ 
  return ( *(int*)a - *(int*)b ); 
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *) realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
      vocab[b].cn = vocab[a].cn;
      vocab[b].word = vocab[a].word;
      b++;
    } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *) "</s>");

  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } 
    else if (i == 0) {
      vocab[i].cn++;
      doc_sizes[corpus_size] = ftell(fin);
      corpus_size++;
      if (corpus_size >= corpus_max_size) {
        printf("[ERROR] Number of documents in corpus larger than \"corpus_max_size\"! Set a larger \"corpus_max_size\" in Line 18 of jose.c!\n");
        exit(1);
      }
    }
    else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void LoadEmb(char *emb_file, double *emb_ptr) {
  long long a, b;
  int *vocab_match_tmp = (int *) calloc(vocab_size, sizeof(int));
  int vocab_size_tmp = 0, word_dim;
  char *current_word = (char *) calloc(MAX_STRING, sizeof(char));
  double *syn_tmp = NULL, norm;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn_tmp, 128, (long long) layer1_size * sizeof(double));
  if (syn_tmp == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }
  printf("Loading embedding from file %s\n", emb_file);
  if (access(emb_file, R_OK) == -1) {
    printf("File %s does not exist\n", emb_file);
    exit(1);
  }
  // read embedding file
  FILE *fp = fopen(emb_file, "r");
  fscanf(fp, "%d", &vocab_size_tmp);
  fscanf(fp, "%d", &word_dim);
  if (layer1_size != word_dim) {
    printf("Embedding dimension incompatible with pretrained file!\n");
    exit(1);
  }
  vocab_size_tmp = 0;
  while (1) {
    fscanf(fp, "%s", current_word);
    a = SearchVocab(current_word);
    if (a == -1) {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &syn_tmp[b]);
    }
    else {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &emb_ptr[a * layer1_size + b]);
      vocab_match_tmp[vocab_size_tmp] = a;
      vocab_size_tmp++;
    }
    if (feof(fp)) break;
  }
  printf("In vocab: %d\n", vocab_size_tmp);
  qsort(&vocab_match_tmp[0], vocab_size_tmp, sizeof(int), IntCompare);
  vocab_match_tmp[vocab_size_tmp] = vocab_size;
  int i = 0;
  for (a = 0; a < vocab_size; a++) {
    if (a < vocab_match_tmp[i]) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        emb_ptr[a * layer1_size + b] = (((next_random & 0xFFFF) / (double) 65536) - 0.5) / layer1_size;
        norm += emb_ptr[a * layer1_size + b] * emb_ptr[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        emb_ptr[a * layer1_size + b] /= sqrt(norm);
    }
    else if (i < vocab_size_tmp) {
      i++;
    }
  }
  fclose(fp);
  free(current_word);
  free(emb_file);
  free(vocab_match_tmp);
  free(syn_tmp);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(double));
  if (syn0 == NULL) {
    printf("Memory allocation failed\n");
    exit(1);
  }

  a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(double));
  a = posix_memalign((void **) &syn1doc, 128, (long long) corpus_size * layer1_size * sizeof(double));
  if (syn1neg == NULL) {
    printf("Memory allocation failed (syn1neg)\n");
    exit(1);
  }
  if (syn1doc == NULL) {
    printf("Memory allocation failed (syn1doc)\n");
    exit(1);
  }

  c_syn0.set_size(layer1_size, L, vocab_size);
  assert(c_syn0.n_slices == vocab_size);
  for (a = 0; a < vocab_size; a++) {
    printf("\rinitializing syn0 |%d|", a);
    arma::mat Y = arma::orth(arma::randn<arma::mat>(layer1_size, L));
    arma::uword r = arma::rank(Y);
    if ((long long)r == L)c_syn0.slice(a) = Y;
    else printf("FULL COLUMN FAIL\n");
  }
  printf("done initializing syn0...\n");

  c_syn1neg.set_size(layer1_size, L, vocab_size);
  assert(c_syn1neg.n_slices == vocab_size);
  for (a = 0; a < vocab_size; a++) {
      printf("\rinitializing syn1neg |%d|", a);
      arma::mat Y = arma::orth(arma::randn<arma::mat>(layer1_size, L));
      arma::uword r = arma::rank(Y);
      if ((long long)r == L)c_syn1neg.slice(a) = Y;
      else printf("FULL COLUMN FAIL\n");
  }
  printf("done initializing syn1neg...\n");
  cout << "corpus size:" << corpus_size << endl;
  c_syn1doc.set_size(layer1_size, P, corpus_size);
  assert(c_syn1doc.n_slices == corpus_size);
  for (a = 0; a < corpus_size; a++) {
      printf("\rinitializing syn1doc |%d|", a);
      arma::mat Y = arma::orth(arma::randn<arma::mat>(layer1_size, P));
      arma::uword r = arma::rank(Y);
      if ((long long)r == P)c_syn1doc.slice(a) = Y;
      else printf("FULL COLUMN FAIL\n");
  }
  printf("done initializing syn1doc...\n");
  
  double norm;
  if (load_emb_file[0] != 0) {
    char *center_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    char *context_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    strcpy(center_emb_file, load_emb_file);
    strcat(center_emb_file, "_w.txt");
    strcpy(context_emb_file, load_emb_file);
    strcat(context_emb_file, "_v.txt");
    LoadEmb(center_emb_file, syn0);
    LoadEmb(context_emb_file, syn1neg);
  }
  else {
    for (a = 0; a < vocab_size; a++) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn1neg[a * layer1_size + b] = (((next_random & 0xFFFF) / (double) 65536) - 0.5) / layer1_size;
        norm += syn1neg[a * layer1_size + b] * syn1neg[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] /= sqrt(norm);
    }
    for (a = 0; a < vocab_size; a++) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (double) 65536) - 0.5) / layer1_size;
        norm += syn0[a * layer1_size + b] * syn0[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        syn0[a * layer1_size + b] /= sqrt(norm);
    }
  }

  for (a = 0; a < corpus_size; a++) {
    norm = 0.0;
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long) 25214903917 + 11;
      syn1doc[a * layer1_size + b] = (((next_random & 0xFFFF) / (double) 65536) - 0.5) / layer1_size;
      norm += syn1doc[a * layer1_size + b] * syn1doc[a * layer1_size + b];
    }
    for (b = 0; b < layer1_size; b++)
      syn1doc[a * layer1_size + b] /= sqrt(norm);
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, doc = 0, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, l3 = 0, c, target, local_iter = iter;
  unsigned long long next_random = (long long) id;
  double f, g, h, step, obj_w = 0, obj_d = 0;
  clock_t now;
  double *neu1 = (double *) calloc(layer1_size, sizeof(double));
  double *grad = (double *) calloc(layer1_size, sizeof(double));
  double *neu1e = (double *) calloc(layer1_size, sizeof(double));
  arma::mat grad_syn0(layer1_size, 1); 
  arma::mat grad_syn1neg(layer1_size, 1);
  arma::mat grad_syn1doc(layer1_size, P);
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);

  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Objective (w): %f  Objective (d): %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, 
               obj_w, obj_d, word_count_actual / (double) (iter * train_words + 1) * 100,
               word_count_actual / ((double) (now - start + 1) / (double) CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (double) (iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      doc = FindLine(fi);
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        if (sample > 0) {
          double ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) /
                     vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (double) 65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }

    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long) num_threads * (long long) id, SEEK_SET);
      continue;
    }

    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long) 25214903917 + 11;
    b = next_random % window;

    for (a = b; a < window * 2 + 1 - b; a++)
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size; // positive center word u
        
        obj_w = 0;
        for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            l3 = word * layer1_size; // positive context word v
          } else {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            target = word_table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            l2 = target * layer1_size; // negative center word u'
            f = 0; h = 0;
            //f = arma::trace(c_syn0.slice(last_word).t()*c_syn1neg.slice(word));
            f = arma::norm(c_syn0.slice(last_word).t()*c_syn1neg.slice(word),"fro"); 
            //h = arma::trace(c_syn0.slice(target).t()*c_syn1neg.slice(word));
            h = arma::norm(c_syn0.slice(target).t()*c_syn1neg.slice(word),"fro");
            if (f - h < margin) {
              obj_w += margin - (f - h);
              // update positive center word
              //arma::Mat<double> grad_context = c_syn0.slice(last_word) - c_syn0.slice(target) - f*c_syn1neg.slice(word) + h*c_syn1neg.slice(word);
              arma::Mat<double> grad_context = c_syn0.slice(last_word)*c_syn0.slice(last_word).t()*c_syn1neg.slice(word)/f - c_syn0.slice(target)*c_syn0.slice(target).t()*c_syn1neg.slice(word)/h;
              arma::Mat<double> proj_grad_context = ortho_proj(grad_context, c_syn1neg.slice(word));
              
              //update positive center word 
              //arma::Mat<double> grad_poscen = c_syn1neg.slice(word) - f*c_syn0.slice(last_word);
              arma::Mat<double> grad_poscen = c_syn1neg.slice(word)*c_syn1neg.slice(word).t()*c_syn0.slice(last_word)/f;
              arma::Mat<double> proj_grad_poscen = ortho_proj(grad_poscen, c_syn0.slice(last_word));
              step = 1 - f;
              c_syn0.slice(last_word) = exp_map(c_syn0.slice(last_word), alpha*proj_grad_poscen, 1.0);
              //c_syn0.slice(last_word) = arma::orth(c_syn0.slice(last_word) + alpha*step*grad_poscen);  
              
              //update negative center word 
              //arma::Mat<double> grad_negcen = h*c_syn0.slice(target) - c_syn1neg.slice(word);
              arma::Mat<double> grad_negcen = -c_syn1neg.slice(word)*c_syn1neg.slice(word).t()*c_syn0.slice(target)/h;
              arma::Mat<double> proj_grad_negcen = ortho_proj(grad_negcen, c_syn0.slice(target));
              //step = 2*h;
              //c_syn0.slice(target) = arma::orth(c_syn0.slice(target) + alpha*step*grad_negcen);
              c_syn0.slice(target) = exp_map(c_syn0.slice(target), alpha*proj_grad_negcen, 1.0);

              //update context word
              step = 1 - (f - h);
              //c_syn1neg.slice(word) = arma::orth(c_syn1neg.slice(word) + alpha*step*grad_context);
              c_syn1neg.slice(word) = exp_map(c_syn1neg.slice(word), alpha*proj_grad_context, 1.0);
            }
          }
        }
      }

    obj_d = 0;
    l1 = doc * layer1_size; // positive document d
    for (d = 0; d < negative + 1; d++) {
      if (d == 0) {
        l3 = word * layer1_size; // positive center word
      } else {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        target = word_table[(next_random >> 16) % table_size];
        if (target == 0) target = next_random % (vocab_size - 1) + 1;
        if (target == word) continue;
        l2 = target * layer1_size; // negative center word u'
        f = 0; h = 0;
        //f = arma::trace(c_syn0.slice(word).t()*c_syn1doc.slice(doc));
        f = arma::norm(c_syn0.slice(word).t()*c_syn1doc.slice(doc),"fro");
        //h = arma::trace(c_syn0.slice(target).t()*c_syn1doc.slice(doc));
        h = arma::norm(c_syn0.slice(target).t()*c_syn1doc.slice(doc),"fro");
        if (f - h < margin) {
          obj_d += margin - (f - h);
          //arma::Mat<double> grad_doc = c_syn0.slice(word) - c_syn0.slice(target) - f*c_syn1doc.slice(doc) + h*c_syn1doc.slice(doc);
          arma::Mat<double> grad_doc = c_syn0.slice(word)*c_syn0.slice(word).t()*c_syn1doc.slice(doc)/f - c_syn0.slice(target)*c_syn0.slice(target).t()*c_syn1doc.slice(doc)/h;
          arma::Mat<double> proj_grad_doc = ortho_proj(grad_doc, c_syn1doc.slice(doc));
          
          //update positive center word 
          //arma::Mat<double> grad_poscen_doc = c_syn1doc.slice(doc) - f*c_syn0.slice(word);
          arma::Mat<double> grad_poscen_doc = c_syn1doc.slice(doc)*c_syn1doc.slice(doc).t()*c_syn0.slice(word)/f;
          arma::Mat<double> proj_grad_poscen_doc = ortho_proj(grad_poscen_doc, c_syn0.slice(word));
          step = 1 - f;
          c_syn0.slice(word) = exp_map(c_syn0.slice(word), alpha*proj_grad_poscen_doc, 1.0);
          //c_syn0.slice(word) = arma::orth(c_syn0.slice(word) + alpha*step*grad_poscen_doc);

          //update negative center word 
          //arma::Mat<double> grad_negcen_doc =  h*c_syn0.slice(target) - c_syn1doc.slice(doc);
          arma::Mat<double> grad_negcen_doc = -c_syn1doc.slice(doc)*c_syn1doc.slice(doc).t()*c_syn0.slice(target)/h;
          arma::Mat<double> proj_grad_negcen_doc = ortho_proj(grad_negcen_doc, c_syn0.slice(target));
          step = 2*h;
          c_syn0.slice(target) = exp_map(c_syn0.slice(target), alpha*proj_grad_negcen_doc, 1.0);
          //c_syn0.slice(target) = arma::orth(c_syn0.slice(target) + alpha*step*grad_negcen_doc);   
          
          //update context word
          step = 1 - (f - h); 
          c_syn1doc.slice(doc) = exp_map(c_syn1doc.slice(doc), alpha*proj_grad_doc, 1.0);
          //c_syn1doc.slice(doc) = arma::orth(c_syn1doc.slice(doc) + alpha*step*grad_doc);
        }
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  free(grad);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c;
  FILE *fo;
  pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);

  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  
  InitNet();
  InitUnigramTable();
  start = clock();
  
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

  if (word_emb[0] != 0) {
    fo = fopen(word_emb, "wb");
    fprintf(fo, "%lld %lld %lld\n", vocab_size, layer1_size, L);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      for (b = 0; b < L; b++) {for (c = 0; c < layer1_size; c++) { if (binary) fwrite(&c_syn0(c, b, a), sizeof(double), 1, fo); 
	else fprintf(fo, "%lf ", c_syn0(c, b, a));
        }}
        //fprintf(fo, "%lf ", c_syn0(c, b, a));
        fprintf(fo, "\n");
    }
    fclose(fo);
  }
  
  if (context_emb[0] != 0) {
    FILE* fa = fopen(context_emb, "wb");
    fprintf(fa, "%lld %lld %lld\n", vocab_size, layer1_size, L);
    for (a = 0; a < vocab_size; a++) {
     fprintf(fa, "%s ", vocab[a].word);
     for(b = 0; b < L; b++) {for (c = 0; c < layer1_size; c++) { fwrite(&c_syn1neg(c, b, a), sizeof(double), 1, fa); }}
        //fprintf(fa, "%lf ", c_syn1neg(c,b,a));
        fprintf(fa, "\n");
    }
    fclose(fa);
  }

  if (doc_output[0] != 0) {
    FILE* fd = fopen(doc_output, "wb");
    fprintf(fd, "%lld %lld %lld\n", corpus_size, layer1_size, P);
    for (a = 0; a < corpus_size; a++) {
      fprintf(fd, "%ld ", a);
      for(b = 0; b < P; b++) for (c = 0; c < layer1_size; c++) fprintf(fd, "%lf ", c_syn1doc(c, b, a));
       fprintf(fd, "\n");
    }
    fclose(fd);
   }
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("Parameters:\n");
    printf("\t-train <file> (mandatory argument)\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-word-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors\n");
    printf("\t-context-output <file>\n");
    printf("\t\tUse <file> to save the resulting word context vectors\n");
    printf("\t-doc-output <file>\n");
    printf("\t\tUse <file> to save the resulting document vectors\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the\n");
    printf("\t\ttraining data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-3)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 2\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads; default is 20\n");
    printf("\t-margin <float>\n");
    printf("\t\tMargin used in loss function to separate positive samples from negative samples; default is 0.15\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations; default is 10\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.04\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-load-emb <file>\n");
    printf("\t\tThe pretrained embeddings will be read from <file>\n");
    printf("\nExamples:\n");
    printf(
        "./jose -train text.txt -word-output jose.txt -size 100 -margin 0.15 -window 5 -sample 1e-3 -negative 2 -iter 10\n\n");
    return 0;
  }
  word_emb[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *) "-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-p", argc, argv)) > 0) P = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-l", argc, argv)) > 0) L = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-load-emb", argc, argv)) > 0) strcpy(load_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *) "-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-word-output", argc, argv)) > 0) strcpy(word_emb, argv[i + 1]);
  if ((i = ArgPos((char *) "-context-output", argc, argv)) > 0) strcpy(context_emb, argv[i + 1]);
  if ((i = ArgPos((char *) "-doc-output", argc, argv)) > 0) strcpy(doc_output, argv[i + 1]);
  if ((i = ArgPos((char *) "-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
  if ((i = ArgPos((char *) "-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  assert(layer1_size > 0); assert(P > 0); assert(L > 0);
  vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
  doc_sizes = (long long *) calloc(corpus_max_size, sizeof(long long));
  if (negative <= 0) {
    printf("ERROR: Nubmer of negative samples must be positive!\n");
    exit(1);
  }
  TrainModel();
  return 0;
}
