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
#include <iostream>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vec.h"
#include <cooperative_groups.h>

using namespace cooperative_groups;


#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
// HACK: THIS IS 1000
#define MAX_SENTENCE_LENGTH 128
#define MAX_CODE_LENGTH 40

const int vocab_hash_size =
    30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word {
    long long cn;
    int *point;
    char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5,
    num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0,
          classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
Vec *syn0, *syn1, *syn1neg;
real *quadform;
real *dev_syn0, *dev_syn1neg, *dev_quadform;
real *dev_gsyn0, *dev_gsyn1neg;
// mask for whether gradient has been updated
bool *dev_mask_syn0;
bool *dev_mask_syn1neg;
real *dev_dots;
real *dev_dots_scratch;


const int NSAMPLES_PER_KERNEL_LAUNCH = 1e4;
int *dev_labels;
char *dev_codes;
unsigned long long *dev_focuses, *dev_ctxes;
unsigned long long *dev_uniq_focuses, *dev_uniq_ctxes;


real *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

real *dev_total_loss;


unsigned long long calcBlockSize(unsigned long long total, unsigned long long thread) {
    if (total / thread == 0) return 1;
    return (total / thread) + (total % thread != 0);
}

__global__ void zeroRealKernel(const int size, real *r) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= size) return;
    r[x] = 0;
}


void InitUnigramTable() {
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    i = 0;
    d1 = pow(vocab[i].cn, power) / train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (double)table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
    
}

// Reads a single word from a file, assuming space + tab + EOL to be word
// boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
    int a = 0, ch;
    while (1) {
        ch = fgetc_unlocked(fin);
        if (ch == EOF) {
            *eof = 1;
            break;
        }
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else
                continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;  // Truncate too long words
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

// Returns position of a word in the vocabulary; if the word is not found,
// returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word))
            return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, char *eof) {
    char word[MAX_STRING], eof_l = 0;
    ReadWord(word, fin, &eof_l);
    if (eof_l) {
        *eof = 1;
        return -1;
    }
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(
            vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    long long l = ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
    if (l > 0) return 1;
    if (l < 0) return -1;
    return 0;
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
        // Words occuring less than min_count times will be discarded from
        // the vocab
        if ((vocab[a].cn < min_count) && (a != 0)) {
            vocab_size--;
            free(vocab[a].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not
            // actual
            hash = GetWordHash(vocab[a].word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = a;
            train_words += vocab[a].cn;
        }
    }
    vocab = (struct vocab_word *)realloc(
        vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    // Allocate memory for the binary tree construction
    for (a = 0; a < vocab_size; a++) {
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
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
        } else
            free(vocab[a].word);
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

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count =
        (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary =
        (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node =
        (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
    for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
    pos1 = vocab_size - 1;
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at
    // a time
    for (a = 0; a < vocab_size - 1; a++) {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) break;
        }
        vocab[a].codelen = i;
        vocab[a].point[0] = vocab_size - 2;
        for (b = 0; b < i; b++) {
            vocab[a].code[i - b - 1] = code[b];
            vocab[a].point[i - b] = point[b] - vocab_size;
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING], eof = 0;
    FILE *fin;
    long long a, i, wc = 0;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab((char *)"</s>");
    while (1) {
        ReadWord(word, fin, &eof);
        if (eof) break;
        train_words++;
        wc++;
        if ((debug_mode > 1) && (wc >= 1000000)) {
            printf("%lldM%c", train_words / 1000000, 13);
            fflush(stdout);
            wc = 0;
        }
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else
            vocab[i].cn++;
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
    for (i = 0; i < vocab_size; i++)
        fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    long long a, i = 0;
    char c, eof = 0;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin, &eof);
        if (eof) break;
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

void InitNet() {
    long long a, b;
    unsigned long long next_random = 1;

    a = posix_memalign((void **)&syn0, 128,
                       (long long)vocab_size * sizeof(Vec));
    if (syn0 == NULL) {
            printf("%d: Memory allocation failed\n", __LINE__);
        exit(1);
    }

    cudaMalloc((void **)&dev_syn0, 
                    (long long) vocab_size * layer1_size * sizeof(real));

    cudaMalloc((void **)&dev_gsyn0, 
                    (long long) vocab_size * layer1_size * sizeof(real));

    zeroRealKernel<<<dim3(calcBlockSize(vocab_size * layer1_size, 1024)), dim3(1024)>>>(vocab_size * layer1_size, dev_gsyn0);

    printf("allocating syn0...");
    for (a = 0; a < vocab_size; ++a) {
        new (Vec)(syn0[a]);
        syn0[a].alloc(layer1_size);
        for (b = 0; b < layer1_size; b++) {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            syn0[a].set(b, (((next_random & 0xFFFF) / (real)65536) - 0.5) / (layer1_size));
        }
        // copy vector to host
        cudaMemcpy(dev_syn0 + layer1_size * a, syn0[a].v, layer1_size *
                        sizeof(real), cudaMemcpyHostToDevice);
    }
    printf("%callocated syn0.\t\t\t\t\n", 13);
    // if (hs) {
    //     a = posix_memalign((void **)&syn1, 128,
    //                        (long long)vocab_size * layer1_size *
    //                        sizeof(real));
    //     if (syn1 == NULL) {
    //         printf("Memory allocation failed\n");
    //         exit(1);
    //     }
    //     for (a = 0; a < vocab_size; a++)
    //         for (b = 0; b < layer1_size; b++) syn1[a * layer1_size + b] = 0;
    // }
    printf("allocating syn1neg...");
    if (negative > 0) {
        a = posix_memalign((void **)&syn1neg, 128,
                           (long long)vocab_size * sizeof(Vec));
        if (syn1neg == NULL) {
            printf("%d: Memory allocation failed\n", __LINE__);
            exit(1);
        }

        for (a = 0; a < vocab_size; a++) {
            new (Vec)(syn1neg[a]);
            syn1neg[a].alloc(layer1_size);
            for (b = 0; b < layer1_size; b++) syn1neg[a].set(b, 0);
        }
    }

    cudaMalloc((void **)&dev_syn1neg, 
                    (long long) vocab_size * layer1_size * sizeof(real));

    zeroRealKernel<<<dim3(calcBlockSize(vocab_size * layer1_size, 1024)), dim3(1024)>>>(vocab_size * layer1_size, dev_syn1neg);

    cudaMalloc((void **)&dev_gsyn1neg, 
                    (long long) vocab_size * layer1_size * sizeof(real));

    printf("%callocated syn1neg.\t\t\t\t\n", 13);



    a = posix_memalign((void **)&quadform, 128,
                    (long long)layer1_size * layer1_size * sizeof(real));

    if (quadform == NULL) {
            printf("%d: Memory allocation failed\n", __LINE__);
            exit(1);
    }
    for(int i = 0; i < layer1_size; ++i){
            for(int j = 0; j < layer1_size; ++j) {
                    // dot product
                    quadform[i * layer1_size + j] = i == j ? 1 : 0;
            }
    }

    cudaMalloc((void **)&dev_quadform, 
                    (long long) layer1_size * layer1_size * sizeof(real));


    cudaMemcpy(dev_quadform, quadform, layer1_size * layer1_size * sizeof(real), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dev_dots, 
                    (long long) NSAMPLES_PER_KERNEL_LAUNCH * sizeof(real));

    cudaMalloc((void **)&dev_labels, 
                    (long long) NSAMPLES_PER_KERNEL_LAUNCH * sizeof(int));

    cudaMalloc((void **)&dev_codes, 
                    (long long) NSAMPLES_PER_KERNEL_LAUNCH * sizeof(char));

    cudaMalloc((void **)&dev_focuses, 
                    (long long) NSAMPLES_PER_KERNEL_LAUNCH * sizeof(unsigned long long));


    cudaMalloc((void **)&dev_ctxes, 
                    (long long) NSAMPLES_PER_KERNEL_LAUNCH * sizeof(unsigned long long));


    cudaMalloc((void **)&dev_uniq_focuses, 
                    (long long) NSAMPLES_PER_KERNEL_LAUNCH * sizeof(unsigned long long));

    cudaMalloc((void **)&dev_uniq_ctxes, 
                    (long long) NSAMPLES_PER_KERNEL_LAUNCH * sizeof(unsigned long long));



    cudaMalloc((void **)&dev_total_loss, (long long) sizeof(real));


    cudaMalloc((void **)&dev_mask_syn0, 
                    (long long) vocab_size * sizeof(bool));

    cudaMalloc((void **)&dev_mask_syn1neg, 
                    (long long) vocab_size * sizeof(bool));


    cudaMalloc((void **)&dev_dots_scratch, 
                    (long long) NSAMPLES_PER_KERNEL_LAUNCH * layer1_size * sizeof(real));

    CreateBinaryTree();
}

inline real sigmoid(real x) {
    // we are trying to calculate sigmoid(127)
    if (x > 5) { return 1; }
    if (x < -5) { return 0; }

    real exp = powf(2, x);
    return exp / (1 + exp);
}


void runkernels(int nsamples, int *labels, 
                unsigned long long *focuses, 
                unsigned long long *ctxes,
                int n_uniq_focuses,
                int n_uniq_ctxes,
                unsigned long long *uniq_focuses,
                unsigned long long *uniq_ctxes) {

        /*
        dim3 threadDims(TX, TY, TZ);
        dim3 blockDims(calcBlockSize(layer1_size, TX),
                calcBlockSize(layer1_size, TY),
                calcBlockSize(nsamples, TZ));
    

        zeroRealKernel<<<dim3(calcBlockSize(NSAMPLES_PER_KERNEL_LAUNCH, 1024)), 
                dim3(1024)>>>(NSAMPLES_PER_KERNEL_LAUNCH, dev_dots);

        zeroRealKernel<<<dim3(calcBlockSize(vocab_size * layer1_size, 1024)), dim3(1024)>>>(vocab_size * layer1_size, dev_gsyn0);
        zeroRealKernel<<<dim3(calcBlockSize(vocab_size * layer1_size, 1024)), dim3(1024)>>>(vocab_size * layer1_size, dev_gsyn1neg);

        // cudaMemset(dev_dots, 1132462080, nsamples * sizeof(real));
        // TODO: do this on the GPU after a syncthreads
        // cudaMemset(dev_mask_syn0, false, vocab_size * sizeof(bool));
        // cudaMemset(dev_mask_syn1neg, true, vocab_size * sizeof(bool));
        // cudaMemset(dev_gsyn0, 1132462080, vocab_size * layer1_size * sizeof(real));
        // cudaMemset(dev_gsyn1neg, 1132462080, vocab_size * layer1_size * sizeof(real));
        // cudaMemset(dev_total_loss, 0, sizeof(real));

        cudaMemcpy(dev_focuses, 
                        focuses, 
                        nsamples * sizeof(unsigned long long), 
                        cudaMemcpyHostToDevice); 
        cudaMemcpy(dev_ctxes, 
                        ctxes, 
                        nsamples * sizeof(unsigned long long), 
                        cudaMemcpyHostToDevice); 

        cudaMemcpy(dev_uniq_focuses, 
                        uniq_focuses, 
                        n_uniq_focuses * sizeof(unsigned long long), 
                        cudaMemcpyHostToDevice); 

        cudaMemcpy(dev_uniq_ctxes, 
                        uniq_ctxes, 
                        n_uniq_ctxes * sizeof(unsigned long long), 
                        cudaMemcpyHostToDevice); 
        cudaMemcpy(dev_labels, 
                        labels, 
                        nsamples * sizeof(int), 
                        cudaMemcpyHostToDevice); 

        assert(n_uniq_ctxes < nsamples);
        assert(n_uniq_focuses < nsamples);

        // printf("fs: %d | ctxes: %d\n", n_uniq_focuses, n_uniq_ctxes);
        const int dimsize = 1;
        dotsHS<<<blockDims, threadDims>>>(layer1_size, nsamples,
                        dev_syn0, dev_quadform, dev_syn1neg, 
                        dev_dots, dev_focuses, dev_ctxes, dimsize);
        grad<<<blockDims, threadDims>>>(layer1_size, nsamples,
                        dev_labels, 
                        dev_dots,
                        dev_syn0, 
                        dev_gsyn0,
                        dev_quadform, 
                        dev_syn1neg,
                        dev_gsyn1neg,
                        alpha,
                        dev_focuses,
                        dev_ctxes,
                        dimsize);
        // backprop<<<blockDims,threadDims>>>(layer1_size, nsamples,
        //                 dev_labels,
        //                 dev_dots,
        //                 dev_syn0,
        //                 dev_gsyn0,
        //                 dev_quadform,
        //                 dev_syn1neg,
        //                 dev_gsyn1neg,
        //                 dev_mask_syn0,
        //                 dev_mask_syn1neg,
        //                 alpha,
        //                 dev_focuses,
        //                 dev_ctxes,
        //                 dimsize);

        
        const int BACKPROPTX = 32;
        const int BACKPROPTY = 4;

        dim3 backpropThreadDims(BACKPROPTX, BACKPROPTY);
        dim3 backpropBlockDims(calcBlockSize(layer1_size, BACKPROPTX),
                calcBlockSize(nsamples, BACKPROPTY));

        backprop2<<<backpropBlockDims,backpropThreadDims>>>(layer1_size, nsamples,
                        dev_labels,
                        dev_dots,
                        dev_syn0,
                        dev_gsyn0,
                        dev_quadform,
                        dev_syn1neg,
                        dev_gsyn1neg,
                        dev_mask_syn0,
                        dev_mask_syn1neg,
                        alpha,
                        dev_focuses,
                        dev_ctxes,
                        n_uniq_focuses,
                        dev_uniq_focuses,
                        n_uniq_ctxes,
                        dev_uniq_ctxes,
                        dimsize);
        // real total_loss;
        // cudaMemcpy(&total_loss, dev_total_loss, sizeof(real), cudaMemcpyDeviceToHost);
        // printf("\ntotal loss: %4.2f\n", total_loss);
        */
}

//x, y = value
//z = data point
const int TX = 32, TY = 32;

__global__ void dotsHS(const int size, const int nsamples,
                const real *syn0,  // LAYER1_SIZE * VOCAB_SIZE
                const real *syn1neg,  // LAYER1_SIZE * VOCAB_SIZE
                real *dotsHS, // dots: [y] NSAMPLES_PER_KERNEL_LAUNCH
                real *dotsScratch, // dotScratch: NSAMPLES_PER_KERNEL_LAUNCH * LAYER1_SIZE
                const unsigned long long *focuses, // NSAMPLER_PER_KERNEL_LAUNCH
                const unsigned long long *ctxes) {


        const unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= size || y >= nsamples) return;



        // dot product of (aT Q b)_xy for sample z
        real dot = syn0[focuses[y] * size + x] * syn1neg[ctxes[y] * size + x];

        dotsScratch[y*size + x] = dot;
        __syncthreads();
        unsigned long long curix = x;
        int partition = size / 2;
        while (partition > 0) {
                if (curix < partition) {
                        __syncthreads();
                        dotsScratch[y*size+curix] += dotsScratch[y*size+partition+curix];
                }
                partition = partition / 2;
        }
        __syncthreads();
        if (curix == 0) {
                atomicAdd(&dotsHS[y], dotsScratch[y * size + 0]);
        }



}

__device__ real sigmoidGPU(real x) {
    // we are trying to calculate sigmoid(127)
    if (x > 5) { return 1; }
    if (x < -5) { return 0; }

    real e = powf(2, x);
    return e / (1 + e);
}


#define FULL_MASK 0xffffffff

__global__ void gradSyn0(const int size, const int nsamples, 
                 const real *dotsHS, const char *codes,
                 real *gsyn0,
                 const real *syn1neg,
                 const real alpha,
                 const unsigned long long *focuses,
                 const unsigned long long *ctxes) {

        const unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= size  || y >= nsamples) { return; }



        // error
        const real err = 1 - codes[y] - sigmoidGPU(dotsHS[y]);
        const real g = err * alpha;

        // all threads that write into the same array index
        atomicAdd(&gsyn0[focuses[y] * size + x], g*syn1neg[ctxes[y]*size + x]);
        // atomicAdd(&syn1neg[ctxes[y] * size + x], g*syn0[focuses[y] * size + x]);

}


__global__ void gradSyn1Neg(const int size, const int nsamples, 
                 const real *dotsHS, const char *codes,
                 const real *syn0,
                 real *syn1neg,
                 const real alpha,
                 const unsigned long long *focuses,
                 const unsigned long long *ctxes) {

        const unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= size  || y >= nsamples) { return; }



        // error
        const real err = 1 - codes[y] - sigmoidGPU(dotsHS[y]);
        const real g = err * alpha;

        // all threads that write into the same array index
        // atomicAdd(&gsyn0[focuses[y] * size + x], g*syn1neg[ctxes[y]*size + x]);
        atomicAdd(&syn1neg[ctxes[y] * size + x], g*syn0[focuses[y] * size + x]);

}




// 2D kernel: layer1_size x nsamples
__global__ void backpropGradIndirect(const int size, const int nseen, 
                 real *vec,
                 real *grad,
                 const unsigned long long *seen) {

        const unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned long long y = blockIdx.y * blockDim.y + threadIdx.y;
        // const int z = blockIdx.z * blockDim.z + threadIdx.z;


        if (x >= size || y >= nseen) { return; }


        atomicAdd(&vec[seen[y]*size + x], grad[seen[y] * size + x]);
        grad[seen[y] * size + x] = 0;

}

void runHSKernel(int nsamples, unsigned long long *focuses, unsigned long long
                *ctxes, char *codes, 
                const unsigned long long num_uniq_focuses, const unsigned long long *uniq_focuses, 
                const unsigned long long num_uniq_ctxes, const unsigned long long *uniq_ctxes) {

        zeroRealKernel<<<dim3(calcBlockSize(NSAMPLES_PER_KERNEL_LAUNCH, 1024)), dim3(1024)>>>(NSAMPLES_PER_KERNEL_LAUNCH, dev_dots);


        // printf("running HS kernel...\n");
        // zeroRealKernel<<<dim3(calcBlockSize(vocab_size * layer1_size, 1024)), dim3(1024)>>>(vocab_size * layer1_size, dev_gsyn0);
        // zeroRealKernel<<<dim3(calcBlockSize(vocab_size * layer1_size, 1024)), dim3(1024)>>>(vocab_size * layer1_size, dev_gsyn1neg);


        cudaMemcpy(dev_focuses, 
                        focuses, 
                        nsamples * sizeof(unsigned long long), 
                        cudaMemcpyHostToDevice); 
        cudaMemcpy(dev_ctxes, 
                        ctxes, 
                        nsamples * sizeof(unsigned long long), 
                        cudaMemcpyHostToDevice); 

        cudaMemcpy(dev_codes, 
                        codes, 
                        nsamples * sizeof(char), 
                        cudaMemcpyHostToDevice); 

        cudaMemcpy(dev_uniq_focuses, 
                        uniq_focuses, 
                        num_uniq_focuses * sizeof(unsigned long long), 
                        cudaMemcpyHostToDevice); 

        cudaMemcpy(dev_uniq_ctxes, 
                        uniq_ctxes, 
                        num_uniq_ctxes * sizeof(unsigned long long), 
                        cudaMemcpyHostToDevice); 

        dim3 threadDims3(TX, TY);
        dim3 blockDims3(calcBlockSize(layer1_size, TX), calcBlockSize(nsamples, TY));

        dotsHS<<<blockDims3, threadDims3>>>(layer1_size, nsamples,
                        dev_syn0, dev_syn1neg, 
                        dev_dots, dev_dots_scratch, dev_focuses, dev_ctxes);

        if (0) {
                real dots[nsamples];
                cudaMemcpy(dots, dev_dots, nsamples * sizeof(real), cudaMemcpyDeviceToHost); 
                printf("DOTS: ");
                for(int i = 0; i < nsamples; ++i)
                        printf("%4.2f ", dots[i]);
                getchar();
        }


        // printf("launching graDHS kernel...\n");
        gradSyn0<<<blockDims3, threadDims3>>>(layer1_size, nsamples,
                        dev_dots, dev_codes,
                        dev_gsyn0,
                        dev_syn1neg, 
                        alpha,
                        dev_focuses, dev_ctxes);

        gradSyn1Neg<<<blockDims3, threadDims3>>>(layer1_size, nsamples,
                        dev_dots, dev_codes,
                        dev_syn0,
                        dev_syn1neg, 
                        alpha,
                        dev_focuses, dev_ctxes);

        if (0) {
                real gsyn0[vocab_size * layer1_size];
                cudaMemcpy(gsyn0, dev_gsyn0, vocab_size * layer1_size * sizeof(real), cudaMemcpyDeviceToHost); 
                printf("gsyn0: ");
                for(int i = 0; i < nsamples; ++i) {
                        for(int j = 0; j < layer1_size; ++j) {
                                printf("%4.2f ", gsyn0[focuses[i] * layer1_size + j]);
                        }
                        printf("\n");
                }
                getchar();
        }




        if (0) {
                real dbg_syn0[vocab_size * layer1_size];
                cudaMemcpy(dbg_syn0, dev_syn0, vocab_size * layer1_size * sizeof(real), cudaMemcpyDeviceToHost); 
                printf("(BEFORE)dbg_syn0: ");
                for(int i = 0; i < nsamples; ++i) {
                        for(int j = 0; j < layer1_size; ++j) {
                                printf("%4.2f ", dbg_syn0[focuses[i] * layer1_size + j]);
                        }
                        printf("\n");
                }
                getchar();
        }

        {
                assert(num_uniq_focuses < NSAMPLES_PER_KERNEL_LAUNCH);
                dim3 threadDims2(TX, TY);
                dim3 blockDims2(calcBlockSize(layer1_size, TX), calcBlockSize(num_uniq_focuses, TY));

                if (0) {
                        std::cout << "blockDims2: (" << blockDims2.x << ", " << blockDims2.y << ", " << blockDims2.z << ")\n";
                        std::cout << "threadDims2: (" << threadDims2.x << ", " << threadDims2.y << ", " << threadDims2.z << ")\n";
                }

                // printf("launching backprophs kernel...\n");
                backpropGradIndirect<<<blockDims2, threadDims2>>>(layer1_size, num_uniq_focuses,
                                dev_syn0, dev_gsyn0, dev_uniq_focuses);
                // printf("ran backprophs kernel...\n");

                if (0) {
                        real dbg_syn0[vocab_size * layer1_size];
                        printf("(AFTER)dbg_syn0: ");
                        cudaMemcpy(dbg_syn0, dev_syn0, vocab_size * layer1_size * sizeof(real), cudaMemcpyDeviceToHost); 
                        for(int i = 0; i < nsamples; ++i) {
                                for(int j = 0; j < layer1_size; ++j) {
                                        printf("%4.2f ", dbg_syn0[focuses[i] * layer1_size + j]);
                                }
                                printf("\n");
                        }
                        getchar();
                }
        }


        // {
        //         assert(num_uniq_ctxes <= NSAMPLES_PER_KERNEL_LAUNCH);
        //         dim3 threadDims2(TX, TY);
        //         dim3 blockDims2(calcBlockSize(layer1_size, TX), calcBlockSize(num_uniq_ctxes, TY));

        //         backpropGradIndirect<<<blockDims2, threadDims2>>>(layer1_size, num_uniq_ctxes,
        //                         dev_syn1neg, dev_gsyn1neg, dev_uniq_ctxes);
        // }

}

void TrainModelThread(void *id) {
    long long a, b, d, word, last_word, sentence_length = 0,
                                        sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label, local_iter = iter;
    unsigned long long next_random = (long long)id;
    char eof = 0;
    real f, err;
    clock_t now;
    real total_loss = 0;
    // real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    // real *neu1e = (real *)calloc(layer1_size, sizeof(real));
    // buffer to store gradient of syn0 in one round
    // real *gsyn0 = (real *)calloc(layer1_size, sizeof(real));
    // buffer to accumulate gradient of syn0
    real *gsyn0_accum = (real *)calloc(layer1_size, sizeof(real));

    // buffer to store gradient of syn1neg
    real *gsyn1neg = (real *)calloc(layer1_size, sizeof(real));
    // buffer to store gradient of syn0 in a train step.
    real *gsyn0 = (real *)calloc(layer1_size, sizeof(real));

    Vec neu1e;
    neu1e.alloczero(layer1_size);

    int ix = 0;
    int labels[NSAMPLES_PER_KERNEL_LAUNCH];
    char codes[NSAMPLES_PER_KERNEL_LAUNCH];

    unsigned long long ctxes[NSAMPLES_PER_KERNEL_LAUNCH],
            focuses[NSAMPLES_PER_KERNEL_LAUNCH],
            uniq_focuses[NSAMPLES_PER_KERNEL_LAUNCH],
            uniq_ctxes[NSAMPLES_PER_KERNEL_LAUNCH];
    int n_uniq_focuses = 0;
    int n_uniq_ctxes = 0;
    bool focus_seen[vocab_size],
        ctx_seen[vocab_size];

    for(int i = 0; i < vocab_size; ++i) focus_seen[i] = false;
    for(int i = 0; i < vocab_size; ++i) ctx_seen[i] = false;

    FILE *fi = fopen(train_file, "rb");
    fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
    while (1) {
        if (word_count - last_word_count > 1) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now = clock();
                printf(
                    "%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: "
                    "%.2fk  Total loss: %4.2f",
                    13, alpha,
                    word_count_actual / (real)(iter * train_words + 1) * 100,
                    word_count_actual / ((real)(now - start + 1) /
                                         (real)CLOCKS_PER_SEC * 1000),
                    total_loss);
                fflush(stdout);
                total_loss = 0;
            }
            /*
            alpha = starting_alpha *
                    (1 - word_count_actual / (real)(iter * train_words + 1));
            if (alpha < starting_alpha * 0.0001)
                alpha = starting_alpha * 0.0001;
            */
        }
        if (sentence_length == 0) {
            while (1) {
                word = ReadWordIndex(fi, &eof);
                if (eof) break;
                if (word == -1) continue;
                word_count++;
                if (word == 0) break;
                // The subsampling randomly discards frequent words while
                // keeping the ranking same
                if (sample > 0) {
                    real ran =
                        (sqrt(vocab[word].cn / (sample * train_words)) + 1) *
                        (sample * train_words) / vocab[word].cn;
                    next_random =
                        next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }
        if (eof || (word_count > train_words / num_threads)) {
            eof = 0;
            word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0) break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            fseek(fi, file_size / (long long)num_threads * (long long)id,
                  SEEK_SET);
            continue;
        }
        word = sen[sentence_position];

        if (word == -1) continue;
        // for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        neu1e.fillzero();
        // for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
        /*
        if (cbow) {  // train the cbow architecture
            // in -> hidden
            cw = 0;
            for (a = b; a < window * 2 + 1 - b; a++)
                if (a != window) {
                    c = sentence_position - window + a;
                    if (c < 0) continue;
                    if (c >= sentence_length) continue;
                    last_word = sen[c];
                    if (last_word == -1) continue;
                    neu1e.accumadd(syn0.ix(last_word));
                    // for (c = 0; c < layer1_size; c++)
                    //     neu1[c] += syn0[c + last_word * layer1_size];
                    cw++;
                }
            if (cw) {
                for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
                if (hs)
                    for (d = 0; d < vocab[word].codelen; d++) {
                        f = 0;
                        l2 = vocab[word].point[d] * layer1_size;
                        // Propagate hidden -> output
                        f += neu1.dot(syn1);
                        // for (c = 0; c < layer1_size; c++)
                        //     f += neu1[c] * syn1[c + l2];
                        if (f <= -MAX_EXP)
                            continue;
                        else if (f >= MAX_EXP)
                            continue;
                        else
                            f = expTable[(int)((f + MAX_EXP) *
                                               (EXP_TABLE_SIZE / MAX_EXP /
        2))];
                        // 'g' is the gradient multiplied by the learning
        rate g = (1 - vocab[word].code[d] - f) * alpha;
                        // Propagate errors output -> hidden
                        for (c = 0; c < layer1_size; c++)
                            neu1e[c] += g * syn1[c + l2];
                        // Learn weights hidden -> output
                        for (c = 0; c < layer1_size; c++)
                            syn1[c + l2] += g * neu1[c];
                    }
                // NEGATIVE SAMPLING
                if (negative > 0)
                    for (d = 0; d < negative + 1; d++) {
                        if (d == 0) {
                            target = word;
                            label = 1;
                        } else {
                            next_random =
                                next_random * (unsigned long
        long)25214903917 + 11; target = table[(next_random >> 16) %
        table_size]; if (target == 0) target = next_random % (vocab_size -
        1) + 1; if (target == word) continue; label = 0;
                        }
                        l2 = target * layer1_size;
                        f = 0;
                        for (c = 0; c < layer1_size; c++)
                            f += neu1[c] * syn1neg[c + l2];
                        if (f > MAX_EXP)
                            g = (label - 1) * alpha;
                        else if (f < -MAX_EXP)
                            g = (label - 0) * alpha;
                        else
                            g = (label - expTable[(int)((f + MAX_EXP) *
                                                        (EXP_TABLE_SIZE /
                                                         MAX_EXP / 2))]) *
                                alpha;
                        for (c = 0; c < layer1_size; c++)
                            neu1e[c] += g * syn1neg[c + l2];
                        for (c = 0; c < layer1_size; c++)
                            syn1neg[c + l2] += g * neu1[c];
                    }
                // hidden -> in
                for (a = b; a < window * 2 + 1 - b; a++)
                    if (a != window) {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        last_word = sen[c];
                        if (last_word == -1) continue;
                        for (c = 0; c < layer1_size; c++)
                            syn0[c + last_word * layer1_size] += neu1e[c];
                    }
            }
        } else {  // train skip-gram
        */
        for (a = b; a < window * 2 + 1 - b; a++)
            if (a != window) {
                c = sentence_position - window + a;
                if (c < 0) continue;
                if (c >= sentence_length) continue;
                last_word = sen[c];
                if (last_word == -1) continue;
                l1 = last_word * layer1_size;

                Vec *syn0v = &syn0[last_word];
                // neu1e.fillzero();
                // for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
                // HIERARCHICAL SOFTMAX
                if (hs) {
                    for (d = 0; d < vocab[word].codelen; d++) {
                        f = 0;
                        l2 = vocab[word].point[d] * layer1_size;
                        const unsigned long long target = vocab[word].point[d];

                        focuses[ix] = last_word;
                        ctxes[ix] = target;
                        codes[ix] = vocab[word].code[d];

                        if (!focus_seen[last_word]) {
                            uniq_focuses[n_uniq_focuses] = last_word;
                            n_uniq_focuses++;
                            focus_seen[last_word] = true;
                        }

                        if (!ctx_seen[target]) {
                            uniq_ctxes[n_uniq_ctxes] = target;
                            n_uniq_ctxes++;
                            ctx_seen[target] = true;
                        }

                        if (ix == NSAMPLES_PER_KERNEL_LAUNCH - 1) {
                                runHSKernel(NSAMPLES_PER_KERNEL_LAUNCH,
                                        focuses,
                                        ctxes,
                                        codes,
                                        n_uniq_focuses,
                                        uniq_focuses,
                                        n_uniq_ctxes,
                                        uniq_ctxes);
                                ix = 0;

                                for(int i = 0; i < n_uniq_ctxes; ++i) {
                                    ctx_seen[uniq_ctxes[i]] = false;
                                }

                                for(int i = 0; i < n_uniq_focuses; ++i) {
                                    focus_seen[uniq_focuses[i]] = false;
                                }
                                n_uniq_ctxes = 0;
                                n_uniq_focuses = 0;

                        } else { ix++; }

                        /*
                        // Propagate hidden -> output
                        float f = 0;
                        for (c = 0; c < layer1_size; c++)
                            f += syn0[c + l1] * syn1[c + l2];
                        if (f <= -MAX_EXP)
                            continue;
                        else if (f >= MAX_EXP)
                            continue;
                        else
                            f = expTable[(int)((f + MAX_EXP) *
                                               (EXP_TABLE_SIZE / MAX_EXP /
                                                2))];
                        // 'g' is the gradient multiplied by the learning
                        // rate
                        float g = (1 - vocab[word].code[d] - f) * alpha;
                        // Propagate errors output -> hidden
                        for (c = 0; c < layer1_size; c++)
                            neu1e[c] += g * syn1[c + l2];
                        // Learn weights hidden -> output
                        for (c = 0; c < layer1_size; c++)
                            syn1[c + l2] += g * syn0[c + l1];
                         */
                    }
                }
                // NEGATIVE SAMPLING
                if (!hs && negative > 0) {
                    for (d = 0; d < negative + 1; d++) {
                        if (d == 0) {
                            target = word;
                            label = 1;
                        } else {
                            next_random =
                                next_random * (unsigned long long)25214903917 +
                                11;
                            target = table[(next_random >> 16) % table_size];
                            if (target == 0)
                                target = next_random % (vocab_size - 1) + 1;
                            if (target == word) continue;
                            label = 0;
                        }

                        labels[ix] = label;
                        focuses[ix] = last_word;
                        ctxes[ix] = target;

                        if (!focus_seen[last_word]) {
                            uniq_focuses[n_uniq_focuses] = last_word;
                            n_uniq_focuses++;
                            focus_seen[last_word] = true;
                        }

                        if (!ctx_seen[target]) {
                            uniq_ctxes[n_uniq_ctxes] = target;
                            n_uniq_ctxes++;
                            ctx_seen[target] = true;
                        }


                        if (ix == NSAMPLES_PER_KERNEL_LAUNCH - 1) {
                                runkernels(NSAMPLES_PER_KERNEL_LAUNCH,
                                        labels,
                                        focuses,
                                        ctxes,
                                        n_uniq_focuses,
                                        n_uniq_ctxes,
                                        uniq_focuses,
                                        uniq_ctxes);


                                ix = 0;
                                n_uniq_ctxes = 0;
                                n_uniq_focuses = 0;
                                for(int i = 0; i < vocab_size; ++i) {
                                    ctx_seen[i] = false;
                                }

                                for(int i = 0; i < vocab_size; ++i) {
                                    focus_seen[i] = false;
                                }
                        } else {
                           ix++;
                        }


                    } // end for loop for negative sampling
                } // end condition around negative samples
                // Learn weights input -> hidden
                // for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];

            } // end a != window

        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }

    assert(ix < NSAMPLES_PER_KERNEL_LAUNCH);

    // consume leftover data.
     runkernels(ix, labels, focuses, ctxes, 
             n_uniq_focuses, n_uniq_ctxes,
             uniq_focuses,
             uniq_ctxes);
    fclose(fi);
    // free(neu1);
    neu1e.freemem();
    // pthread_exit(NULL);
}

void TrainModel() {
    long a, b, c, d;
    FILE *fo;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    printf("Starting training using file %s\n", train_file);
    starting_alpha = alpha;
    if (read_vocab_file[0] != 0)
        ReadVocab();
    else
        LearnVocabFromTrainFile();
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;
    InitNet();
    if (negative > 0) InitUnigramTable();
    start = clock();
    if (iter > 0) {
            TrainModelThread((void *)0);
            // for (a = 0; a < num_threads; a++)
            //         pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
            // for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    }
    fo = fopen(output_file, "wb");
    if (classes == 0) {
        real syn0_out[vocab_size * layer1_size];
        cudaMemcpy(syn0_out, dev_syn0, vocab_size * layer1_size * sizeof(real), cudaMemcpyDeviceToHost);
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        printf("%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++) {
            fprintf(fo, "%s ", vocab[a].word);
            printf("\n%s ", vocab[a].word);


            if (binary) {
                for (b = 0; b < layer1_size; b++) {
                    fwrite(syn0_out + a *layer1_size + b, sizeof(real), 1, fo);
                    printf("%f ", *(syn0_out + a *layer1_size + b));
                }
            } else {
                for (b = 0; b < layer1_size; b++) {
                    fprintf(fo, "%lf ", syn0[a].ix(b));
                }
            }
            fprintf(fo, "\n");
        }
    } else {
        // Run K-means on the word vectors
        int clcn = classes, iter = 10, closeid;
        int *centcn = (int *)malloc(classes * sizeof(int));
        int *cl = (int *)calloc(vocab_size, sizeof(int));
        real closev, x;
        real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
        for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
        for (a = 0; a < iter; a++) {
            for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
            for (b = 0; b < clcn; b++) centcn[b] = 1;
            for (c = 0; c < vocab_size; c++) {
                for (d = 0; d < layer1_size; d++)
                    cent[layer1_size * cl[c] + d] += syn0[c].ix(d);
                centcn[cl[c]]++;
            }
            for (b = 0; b < clcn; b++) {
                closev = 0;
                for (c = 0; c < layer1_size; c++) {
                    cent[layer1_size * b + c] /= centcn[b];
                    closev +=
                        cent[layer1_size * b + c] * cent[layer1_size * b + c];
                }
                closev = sqrt(closev);
                for (c = 0; c < layer1_size; c++)
                    cent[layer1_size * b + c] /= closev;
            }
            for (c = 0; c < vocab_size; c++) {
                closev = -10;
                closeid = 0;
                for (d = 0; d < clcn; d++) {
                    x = 0;
                    for (b = 0; b < layer1_size; b++)
                        x += cent[layer1_size * d + b] * syn0[c].ix(b);
                    if (x > closev) {
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;
            }
        }
        // Save the K-means classes
        for (a = 0; a < vocab_size; a++)
            fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
        free(centcn);
        free(cent);
        free(cl);
    }
    fclose(fo);
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

int mainw2v(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf(
            "\t\tUse <file> to save the resulting word vectors / word "
            "clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf(
            "\t\tSet threshold for occurrence of words. Those that appear "
            "with "
            "higher frequency in the training data\n");
        printf(
            "\t\twill be randomly down-sampled; default is 1e-3, useful "
            "range "
            "is (0, 1e-5)\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
        printf("\t-negative <int>\n");
        printf(
            "\t\tNumber of negative examples; default is 5, common values "
            "are "
            "3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf(
            "\t\tThis will discard words that appear less than <int> "
            "times; "
            "default is 5\n");
        printf("\t-alpha <float>\n");
        printf(
            "\t\tSet the starting learning rate; default is 0.025 for "
            "skip-gram and 0.05 for CBOW\n");
        printf("\t-classes <int>\n");
        printf(
            "\t\tOutput word classes rather than word vectors; default "
            "number "
            "of classes is 0 (vectors are written)\n");
        printf("\t-debug <int>\n");
        printf(
            "\t\tSet the debug mode (default = 2 = more info during "
            "training)\n");
        printf("\t-binary <int>\n");
        printf(
            "\t\tSave the resulting vectors in binary moded; default is 0 "
            "(off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf(
            "\t\tThe vocabulary will be read from <file>, not constructed "
            "from "
            "the training data\n");
        printf("\t-cbow <int>\n");
        printf(
            "\t\tUse the continuous bag of words model; default is 1 (use "
            "0 "
            "for skip-gram model)\n");
        printf("\nExamples:\n");
        printf(
            "./word2vec -train data.txt -output vec.txt -size 200 -window "
            "5 "
            "-sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }
    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0)
        layer1_size = atoi(argv[i + 1]);
    fprintf(stdout, "size: %lld\n", layer1_size);
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0)
        strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0)
        strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0)
        strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0)
        debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
        binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
    fprintf(stdout, "cbow: %d\n", cbow);

    if (cbow) alpha = 0.05;
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0)
        alpha = atof(argv[i + 1]);
    fprintf(stdout, "alpha: %f\n", alpha);

    if ((i = ArgPos((char *)"-output", argc, argv)) > 0)
        strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
        window = atoi(argv[i + 1]);
    fprintf(stdout, "window: %d\n", window);

    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0)
        sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
    fprintf(stdout, "hs: %d\n", hs);

    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0)
        negative = atoi(argv[i + 1]);
    fprintf(stdout, "negative: %d\n", negative);

    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
        num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
        min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-classes", argc, argv)) > 0)
        classes = atoi(argv[i + 1]);
    vocab =
        (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) *
                          MAX_EXP);  // Precompute the exp() table
        expTable[i] =
            expTable[i] / (expTable[i] + 1);  // Precompute f(x) = x / (x + 1)
    }
    printf("INT: %d\n", float(0.0));
    printf("FLOAT: %f\n", 1132462080);
    TrainModel();
    return 0;
}

int main(int argc, char *argv[]) {
    mainw2v(argc, argv);
    return 0;
}
