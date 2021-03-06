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

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

// #define DEBUG_ANGLE2VEC
// #define EXPENSIVE_CHECKS

pthread_mutex_t mutex_syn0 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_syn1neg = PTHREAD_MUTEX_INITIALIZER;

const int vocab_hash_size =
    30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;  // Precision of float numbers

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
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;


void angleprecompute(int n, const real theta[n-1], real coss[n-1], real
        sins[n-1], real sinaccum[n-1][n-1]);


void angle2vec(int n, const real coss[n - 1], const real sins[n - 1], const
        real sinaccum[n-1][n-1], real out[n]);


real uniform01(long long unsigned int* next_random){
    *next_random =
        *next_random * (unsigned long long)25214903917 + 11;
    return (*next_random & 0xFFFF) / ((real) 65536);
}

real uniformsign(long long unsigned int *next_random) {
    return uniform01(next_random) >= 0.5 ? 1 : -1;
}

// gaussian distribution, mean 0, variance 1
real standardNormal(long long unsigned int *next_random) {

    // https://en.wikibooks.org/wiki/Statistics/Distributions/Uniform
    // mean of standard normal = sum of means of iid = 0 * NSAMPLES = 0
    // stddev of standard normal = sum of stddevs of iid = 0 * 12 = 0
    real sum = 0;
    const int NSAMPLES = 10;
    // var = (1 - (-1))^2 / 12 = 2^2 / 12 = 4 / 12 = 1 / 3
    // stddev = sqrt(1/3)
    const real uniformStddev = sqrtf(1.0 / 3);

    for(int i = 0; i < NSAMPLES; ++i) {
        // uniform in [-1, 1]
        sum += uniformsign(next_random) * uniform01(next_random);
    }

    // convert gaussian X into standard normal: (X - mu) / sigma
    return (sum - 0.0) / 2 * (uniformStddev * NSAMPLES);
}

void sampleRandomPointSphere(int n, real angles[n-1], long long unsigned int *next_random) {
    real vec[n];
    float lensq = 0;
    for(int i = 0; i < n; ++i) {
        vec[i] = standardNormal(next_random);
        lensq += vec[i] * vec[i];
    }

    const float len = sqrt(lensq);
    for(int i = 0; i < n; ++i) {
        vec[i] /= len;
    };
    
    // n = 5
    // x0 = cos t0
    // x1 = sin t0 cos t1
    // x2 = sin t0 sin t1 cos t2
    // x3 = sin t0 sin t1 sin t2 cos t3
    // x4 = sin t0 sin t1 sin t2 sin t3
    //
    // x4/x3 = sin t3 / cos t3 = tan t3
    angles[n-2] = atan2(vec[n-1], vec[n-2]);
    // to compute t2, we need to take atan2(x3, x2 * cos(t3))
    for(int i = n - 3; i >= 0; i--) {
        angles[i] =  atan2(vec[i+1], vec[i] * cos(angles[i+1]));
    }

    #ifdef EXPENSIVE_CHECKS
    real sins[n-1], coss[n-1], sinaccum[n-1][n-1];
    real vecinv[n];
    angleprecompute(n, angles, coss, sins, sinaccum);
    angle2vec(n, coss, sins, sinaccum, vecinv);

    float lensq_vecinv = 0;
    for(int i = 0; i < n; ++i) {
        lensq_vecinv += vecinv[i] * vecinv[i];
    }

    assert (fabs(lensq_vecinv - 1) < 1e-2);

    //TODO: find out why we have sign differences.
    for(int i = n - 1; i >= 0; --i) {
        if (fabs(fabs(vecinv[i]) - fabs(vec[i])) > 1e-2) {
            printf("mismatch(n=%d): expected[%d] = %f | found[%d] = %f\n",  n,
                    i, vec[i], i, vecinv[i]);
            assert(0 && "mismatch in expected and recovered vector");
        }
    }
    #endif

}

// code from:
// https://stackoverflow.com/questions/11261170/c-and-maths-fast-approximation-of-a-trigonometric-function
/* not quite rint(), i.e. results not properly rounded to nearest-or-even */
real my_rint (real x)
{
    real t = floor (fabs(x) + 0.5);
    return (x < 0.0) ? -t : t;
}

/* minimax approximation to cos on [-pi/4, pi/4] with rel. err. ~= 7.5e-13 */
real cos_core (real x)
{
    real x8, x4, x2;
    x2 = x * x;
    x4 = x2 * x2;
    x8 = x4 * x4;
    /* evaluate polynomial using Estrin's scheme */
    return (-2.7236370439787708e-7 * x2 + 2.4799852696610628e-5) * x8 +
        (-1.3888885054799695e-3 * x2 + 4.1666666636943683e-2) * x4 +
        (-4.9999999999963024e-1 * x2 + 1.0000000000000000e+0);
}

/* minimax approximation to sin on [-pi/4, pi/4] with rel. err. ~= 5.5e-12 */
real sin_core (real x)
{
    real x4, x2;
    x2 = x * x;
    x4 = x2 * x2;
    /* evaluate polynomial using a mix of Estrin's and Horner's scheme
     * */
    return ((2.7181216275479732e-6 * x2 - 1.9839312269456257e-4) * x4 + 
            (8.3333293048425631e-3 * x2 - 1.6666666640797048e-1)) * x2 * x + x;
}

/* relative error < 7e-12 on [-50000, 50000] */
real sin_fast (real x)
{
    real q, t;
    int quadrant;
    /* Cody-Waite style argument reduction */
    q = my_rint (x * 6.3661977236758138e-1);
    quadrant = (int)q;
    t = x - q * 1.5707963267923333e+00;
    t = t - q * 2.5633441515945189e-12;
    if (quadrant & 1) {
        t = cos_core(t);
    } else {
        t = sin_core(t);
    }
    return (quadrant & 2) ? -t : t;
}


real cos_fast(real x) {
    return sin_fast(M_PI / 2.0 + x);
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
        // Words occuring less than min_count times will be discarded from the
        // vocab
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
    // Following algorithm constructs the Huffman tree by adding one node at a
    // time
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
                       (long long)vocab_size * (layer1_size -1) * sizeof(real));
    if (syn0 == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    if (hs) {
        a = posix_memalign((void **)&syn1, 128,
                           (long long)vocab_size * layer1_size * sizeof(real));
        if (syn1 == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < vocab_size; a++)
            for (b = 0; b < layer1_size; b++) syn1[a * layer1_size + b] = 0;
    }
    if (negative > 0) {
        a = posix_memalign((void **)&syn1neg, 128,
                           (long long)vocab_size * (layer1_size -1) * sizeof(real));
        if (syn1neg == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < vocab_size; a++) {
            sampleRandomPointSphere(layer1_size, syn1neg + a * (layer1_size - 1), &next_random);
        }

    }
    for (a = 0; a < vocab_size; a++) {
            sampleRandomPointSphere(layer1_size, syn0 + a * (layer1_size - 1), &next_random);
    }
    CreateBinaryTree();
}

// given angles, precompute sin(theta_i), cos(theta_i) and 
//  sin(theta_i) * sin(theta_{i+1}) *  ... * sin(theta_j) 0 <= i, j <= n-1
void angleprecompute(const int n, const real theta[n-1], real coss[n-1], 
        real sins[n-1], real sinaccum[n-1][n-1]) {
    for(int i = 0; i < n - 1; i++) {
        coss[i] = cos(theta[i]);
        sins[i] = sin(theta[i]);
        // cos^2 x + sin^2 x = 1
        int safe =  fabs(1.0 - (coss[i] * coss[i] + sins[i] * sins[i])) < 1e-2;
        if (!safe) {
            printf("theta: %f | real:%f / coss: %f | real: %f / sins: %f\n", theta[i], 
                    cos(theta[i]), coss[i], sin(theta[i]), sins[i]);
            assert(0);
        }
    }
    
    // check interval [i..j]
    for(int i = 0; i < n - 1; ++i) {
        // j < i
        for(int j = 0; j < i; ++j) { sinaccum[i][j] = 1; }
        //j = i
        sinaccum[i][i] = sins[i];
        // j > 1
        for(int j = i + 1; j < n - 1; ++j) {
            sinaccum[i][j] = sins[j] * sinaccum[i][j-1];
        }
    }
}

// convert angles to vectors for a given index
void angle2vec(const int n, const real coss[n - 1], const real sins[n - 1], const real sinaccum[n-1][n-1],
        real out[n]) {

    // reference
    // x1          = c1
    // x2          = s1 c2
    // x3          = s1 s2 c3
    // x4          = s1 s2 s3 c4
    // x5          = s1 s2 s3 s4 c5
    // x6 = xfinal = s1 s2 s3 s4 s5
    for(int i = 0; i < n; i++) {
        out[i] = (i == 0 ? 1 : sinaccum[0][i-1]) * (i == n-1 ? 1 : coss[i]);
    }


    #ifdef EXPENSIVE_CHECKS
    real lensq = 0;
    for(int i = 0; i < n; i++) {
        lensq += out[i] * out[i];
    }
    if(fabs(lensq - 1) >= 0.2) { 
        printf("lensq: %f\n", lensq);
        printf("  cos: ["); 
        for(int i = 0; i < n; ++i) {
            printf("%f ", coss[i]);
        }
        printf("]\n"); 
        printf("  sin: ["); 
        for(int i = 0; i < n; ++i) {
            printf("%f ", sins[i]);
        }
        printf("]\n"); 
        printf("  vec: ["); 
        for(int i = 0; i < n; ++i) {
            printf("%f ", out[i]);
        }
        printf("]\n"); 
    }
    assert(fabs(lensq - 1) < 0.2);
    #endif
}

void debugPrintAngleRepr(int n, int derix, int vecix) {
    printf("  d/d%d[", derix);

    for(int i = 0; i < vecix; ++i){
        printf("sin(%d)", i);
    }
    if (vecix != n - 1) {
        printf("cos(%d)", vecix);
    }
    printf("]");
}

// x0 = c0               v0
// x1 = s0 c1            v1
// x2 = s0 s1 c2         v2
// x3 = s0 s1 s2 c3      v3
// x4 = s0 s1 s2 s3 c4   v4
// x5 = s0 s1 s2 s3 s4   v5
// compute: d/dtheta(xindex)
real angle2derTerm(const int n, const int theta, const int xindex, const real coss[n-1], const real sins[n-1],
        const real sinprods[n-1][n-1], const real vec[n], const real g) {
    #ifdef EXPENSIVE_CHECKS
    assert(xindex >= 0 && xindex <= n - 1);
    assert(theta >= 0 && theta <= n - 2);
    #endif
    // term n contains thetas of {0..n}
    if (xindex < theta) { return 0; }

    // final term
    // need to take the derivative of
    // xn = s0 s1 ... sn
    // d/di(s0 s1 ... sn) = s0 s1 ... s{i-1} ci s{i+1} ... sn
    if (xindex == n-1) {
        const real lprod = theta == 0 ? 1 : sinprods[0][theta - 1];
        const real rprod = theta == n - 2 ? 1 : sinprods[theta+1][n-2];
        return lprod * coss[theta] * rprod * g * vec[xindex];
    }

    // No more final term. Term is of the form
    // xn = s0 s1 s2 s3 ... s{n-1} cn 
    // Two cases:
    // case 1. d/dn(xn) = d/dn(s0 s1 ...s{n-1} cn) = s0 s1...s{n-1} |(- sn)|
    // case 2. d/di(xn) = d/dn(s0 s1 ...s{n-1} cn) = s0 s1..s{i-1} |ci| s{i+1}..s{n-1}..cn
    if (theta == xindex) { // case 1
        const real lprod = theta == 0 ? 1 : sinprods[0][theta-1];
        return lprod * (-sins[theta]) * g * vec[xindex];
    } else { // case 2
        assert(theta < xindex);
        const real lprod = theta == 0 ? 1 : sinprods[0][theta - 1];
        const real rprod = sinprods[theta+1][xindex-1];
        return lprod * coss[theta] * rprod * coss[xindex] * g * vec[xindex];
    }

    assert(0 && "unreachable");
}

// store in out[i] the derivative of d(angle2vec(thetas) . vec)/d(theta_i)
// NOTE: does not zero out ders
void angle2der(const int n, const real coss[n - 1], const real sins[n - 1],  
        const real sinprods[n-1][n-1], const real vec[n], const real g, real ders[n - 2]) {
    // x0 = c0               v0
    // x1 = s0 c1            v1
    // x2 = s0 s1 c2         v2
    // x3 = s0 s1 s2 c3      v3
    // x4 = s0 s1 s2 s3 c4   v4
    // x5 = s0 s1 s2 s3 s4   v5
    for(int xindex = 0; xindex < n; xindex++) {
        for(int theta = 0; theta < n - 1; theta++) {
            ders[theta] += angle2derTerm(n, theta, xindex, coss, sins, sinprods, vec, g);
        }
    }
}

void *TrainModelThread(void *id) {
    long long a, b, d, cw, word, last_word, sentence_length = 0,
                                            sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label, local_iter = iter;
    unsigned long long next_random = (long long)id;
    char eof = 0;
    real f, g;
    clock_t now;
    real *neu1 = (real *)calloc(layer1_size, sizeof(real));
    // real *neu1e = (real *)calloc(layer1_size - 1, sizeof(real));
    real neu1e[layer1_size-1];
    // real *syn0vec = (real *)calloc(layer1_size, sizeof(real));
    real syn0vec[layer1_size];
    real syn1vec[layer1_size];
    //real *syn0cos = (real *)calloc(layer1_size-1, sizeof(real));
    real syn0cos[layer1_size - 1];
    real syn1cos[layer1_size - 1];
    //real *syn0sin = (real *)calloc(layer1_size-1, sizeof(real));
    real syn0sin[layer1_size - 1];
    real syn1sin[layer1_size - 1];
    // syn0sinaccum[n + (layer_size - 1) * m] = product of syn0sin in range [n, m] (inclusive)
    // real *syn0sinaccum = (real *)calloc((layer1_size -1)* (layer1_size-1), sizeof(real));
    real syn0sinaccum[layer1_size-1][layer1_size-1];
    real syn1sinaccum[layer1_size-1][layer1_size-1];
    real total_loss = 0;

    FILE *fi = fopen(train_file, "rb");
    fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
    while (1) {
        if (word_count - last_word_count > 100) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            if ((debug_mode > 1)) {
                now = clock();
                printf(
                    "%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk Total loss: %.7f ",
                    13, alpha,
                    word_count_actual / (real)(iter * train_words + 1) * 100,
                    word_count_actual / ((real)(now - start + 1) /
                                         (real)CLOCKS_PER_SEC * 1000),
                    total_loss);
            total_loss = 0;
                fflush(stdout);
            }

            alpha = starting_alpha *
                    (1 - word_count_actual / (real)(iter * train_words + 1));
            if (alpha < starting_alpha * 0.001)
                alpha = starting_alpha * 0.001;

            if (alpha < 0.001) alpha = 0.001;
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
            printf("done with iteration (%d) (thread %d)\n", local_iter, (int)id);
            word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0) break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            eof = 0;
            fseek(fi, file_size / (long long)num_threads * (long long)id,
                  SEEK_SET);
            continue;
        }
        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size - 1; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;
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
                    for (c = 0; c < layer1_size; c++)
                        neu1[c] += syn0[c + last_word * layer1_size];
                    cw++;
                }
            if (cw) {
                for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
                if (hs)
                    for (d = 0; d < vocab[word].codelen; d++) {
                        f = 0;
                        l2 = vocab[word].point[d] * layer1_size;
                        // Propagate hidden -> output
                        for (c = 0; c < layer1_size; c++)
                            f += neu1[c] * syn1[c + l2];
                        if (f <= -MAX_EXP)
                            continue;
                        else if (f >= MAX_EXP)
                            continue;
                        else
                            f = expTable[(int)((f + MAX_EXP) *
                                               (EXP_TABLE_SIZE / MAX_EXP / 2))];
                        // 'g' is the gradient multiplied by the learning rate
                        g = (1 - vocab[word].code[d] - f) * alpha;
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
                                next_random * (unsigned long long)25214903917 +
                                11;
                            target = table[(next_random >> 16) % table_size];
                            if (target == 0)
                                target = next_random % (vocab_size - 1) + 1;
                            if (target == word) continue;
                            label = 0;
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
            for (a = b; a < window * 2 + 1 - b; a++)
                if (a != window) {
                    c = sentence_position - window + a;
                    if (c < 0) continue;
                    if (c >= sentence_length) continue;
                    last_word = sen[c];
                    if (last_word == -1) continue;
                    l1 = last_word * (layer1_size - 1);

                    // zero out gradient buffer for current word.
                    for (c = 0; c < layer1_size-1; c++) neu1e[c] = 0;
                    // HIERARCHICAL SOFTMAX
                    if (hs)
                        for (d = 0; d < vocab[word].codelen; d++) {
                            f = 0;
                            l2 = vocab[word].point[d] * (layer1_size - 1);
                            // Propagate hidden -> output
                            for (c = 0; c < layer1_size; c++)
                                f += syn0[c + l1] * syn1[c + l2];
                            if (f <= -MAX_EXP)
                                continue;
                            else if (f >= MAX_EXP)
                                continue;
                            else
                                f = expTable[(
                                    int)((f + MAX_EXP) *
                                         (EXP_TABLE_SIZE / MAX_EXP / 2))];
                            // 'g' is the gradient multiplied by the learning
                            // rate
                            g = (1 - vocab[word].code[d] - f) * alpha;
                            // Propagate errors output -> hidden
                            for (c = 0; c < layer1_size; c++)
                                neu1e[c] += g * syn1[c + l2];
                            // Learn weights hidden -> output
                            for (c = 0; c < layer1_size; c++)
                                syn1[c + l2] += g * syn0[c + l1];
                        }
                    // NEGATIVE SAMPLING
                    if (negative > 0) {
                        pthread_mutex_lock(&mutex_syn0);
                        angleprecompute(layer1_size, syn0 + l1, syn0cos,
                                syn0sin, syn0sinaccum);
                        pthread_mutex_unlock(&mutex_syn0);

                        // printf("anglevec: syn0\n");
                        angle2vec(layer1_size, syn0cos, syn0sin, syn0sinaccum, syn0vec);

                        /*
                        // initialize sin/cos tables for syn0
                        for(c = 0; c < layer1_size - 1; c++) {
                            syn0cos[c] = cos(syn0[c + l1]);
                            syn0sin[c] = sin(syn0[c + l1]);
                        }

                        // initialize sin[n,m] = product of sin[n] * sin[n+1] * ... * sin[m]
                        // tables
                        for(int n = 0; n < layer1_size - 2; n++) {
                            // [n, n] = sin[n]
                            syn0sinaccum[n][n] = syn0sin[n];
                            for(int m = n - 1; m >= 0; m--) {
                                // [m, n] = sin[n] * [n+1, m]
                                syn0sinaccum[m][n] = 
                                    syn0sin[m] * 
                                    syn0sinaccum[m+1][n];
                            }
                            //[m, n] where m > n
                            for(int m = n + 1; m < layer1_size - 1; m++) {
                                syn0sinaccum[m][n] = 1;
                            }
                        }
                        */

                        for (d = 0; d < negative + 1; d++) {
                            if (d == 0) {
                                target = word;
                                label = 1;
                            } else {
                                next_random =
                                    next_random *
                                        (unsigned long long)25214903917 +
                                    11;
                                target =
                                    table[(next_random >> 16) % table_size];
                                if (target == 0)
                                    target = next_random % (vocab_size - 1) + 1;
                                if (target == word) continue;
                                label = 0;
                            }
                            l2 = target * (layer1_size - 1);

                            pthread_mutex_lock(&mutex_syn1neg);
                            angleprecompute(layer1_size, syn1neg + l2, syn1cos,
                                    syn1sin, syn1sinaccum);
                            pthread_mutex_unlock(&mutex_syn1neg);
                            angle2vec(layer1_size, syn1cos, syn1sin, syn1sinaccum, syn1vec);


                            // compute dot product
                            f = 0;
                            for (c = 0; c < layer1_size; c++) {
                                //dot product is between -1 and 1
                                f += 5.0 * syn0vec[c] * syn1vec[c];
                            }
                            // ----
                            // loss = (label - syn0 . syn1)^2
                            // dloss/dxi = 2 (label - syn0 . syn1) *
                            //                 d(syn.syn1)/dxi
                            // ----

                            // loss = (label - sigmoid (syn0 . syn1))^2
                            // gradient = d(loss) = 2 . (label - sigmoid(syn0 . syn1)) d (syn0 . syn1)
                            // if (f > MAX_EXP)
                            //     g = (label - 1) * alpha;
                            // else if (f < -MAX_EXP)
                            //     g = (label - 0) * alpha;
                            // else
                            //     g = (label - expTable[(int)((f + MAX_EXP) *
                            //                 (EXP_TABLE_SIZE /
                            //                  MAX_EXP / 2))]) * alpha;
                            g = (label - f) * alpha * (label == 0 ? 10:1);
                            total_loss += g * g;
                            
                            // buffer gradients of focus
                            angle2der(layer1_size, syn0cos,
                                    syn0sin, syn0sinaccum,
                                    syn1vec, g * alpha, neu1e);


                            // write the gradients of context into the vector
                            // pthread_mutex_lock(&mutex_syn1neg);
                            pthread_mutex_lock(&mutex_syn1neg);
                            angle2der(layer1_size, syn1cos,
                                    syn1sin, syn1sinaccum,
                                    syn0vec, g * alpha, syn1neg + l2);
                            // pthread_mutex_unlock(&mutex_syn1neg);
                            pthread_mutex_unlock(&mutex_syn1neg);

                        } // end negative samples loop

                        // Learn weights input -> hidden
                        pthread_mutex_lock(&mutex_syn0);
                        for (c = 0; c < layer1_size-1; c++) syn0[c + l1] += neu1e[c];
                        pthread_mutex_unlock(&mutex_syn0);
                    } // end negative sampling if condition
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
    pthread_exit(NULL);
}

void TrainModel() {
    long a, b, c, d;
    FILE *fo;
    real *syn0vec = (real *)calloc(layer1_size, sizeof(real));
    real coss[layer1_size - 1];
    real sins[layer1_size - 1];
    real sinaccum[layer1_size - 1][layer1_size - 1];

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
    // if number of iterations is > 0, then run training
    if (iter > 0) {
        for (a = 0; a < num_threads; a++)
            pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    }
    fo = fopen(output_file, "wb");
    if (classes == 0) {
        // Save the word vectors
        fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
        for (a = 0; a < vocab_size; a++) {
            fprintf(fo, "%s ", vocab[a].word);
            angleprecompute(layer1_size, syn0 + a * (layer1_size - 1), coss, sins, sinaccum);
            angle2vec(layer1_size, coss, sins, sinaccum, syn0vec);

            if (binary)
                for (b = 0; b < layer1_size; b++)
                    fwrite(&syn0vec[b], sizeof(real), 1, fo);
            else
                for (b = 0; b < layer1_size; b++)
                    fprintf(fo, "%lf ", syn0vec[b]);
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
                    cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
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
                        x += cent[layer1_size * d + b] *
                             syn0[c * layer1_size + b];
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

// given angle inputs, will print derivative outputs.
// first provide 4 vector, then provide 3 angles
void test(int argc, char **argv) {
    int ix = 2; // word2vec -stress-test
    int n = atoi(argv[ix++]);
    float v[n];
    printf("v: ");
    for(int i = 0; i < n; ++i) {
        v[i] = atoi(argv[ix++]);
        printf("%f ", v[i]);
    }
    printf("\n");



    printf("angles: ");
    float angles[n-1];
    for(int i = 0; i < n - 1; ++i) {
        angles[i] = atoi(argv[ix++]);
        printf("%f ", angles[i]);
    }
    printf("\n");
    float sins[n-1];
    float coss[n-1];
    float sinaccum[n-1][n-1];

    angleprecompute(n, angles, coss, sins, sinaccum);

    printf("sins: ");
    for(int i = 0; i < n - 1; ++i) {
        printf("%f ", sins[i]);
    }
    printf("\n");

    printf("coss: ");
    for(int i = 0; i < n - 1; ++i) {
        printf("%f ", coss[i]);
    }
    printf("\n");
    // check interval [i..j]
    for(int i = 0; i < n - 1; ++i) {
        for(int j = 0; j < n - 1; ++ j) {
            float prod = 1;
            for(int k = i; k <= j; ++k) {
                prod *= sin(angles[k]);
            }
            if(fabs(sinaccum[i][j] - prod) > 1e-2) {
                printf("i: %d | j: %d | sinaccum[i][j]: %f | prod: %f",  i, j, sinaccum[i][j], prod);
                assert(0 && "incorrect value in sinaccum");
            }
        }
    }

    float angles_vec[n];
    angle2vec(n, coss, sins, sinaccum, angles_vec);
    printf("angle2vec: ");
    for(int i = 0; i < n; i++) { printf("%f ", angles_vec[i]); } 
    printf("\n");

    float angles_der[n-1];
    for(int i = 0; i < n - 1; ++i) { angles_der[i] = 0; }

    angle2der(n, coss,
            sins, sinaccum,
            v, 1, angles_der);
    
    printf("angles_der: ");
    for(int i = 0; i < n - 1; ++i) {
        printf("%f ", angles_der[i]);
    }
    printf("\n");

    // test gaussian code
    {
        long long unsigned int  next_random = 1;
        float random_angle[n-1];
        sampleRandomPointSphere(n, random_angle, &next_random);
    }

}

int main(int argc, char **argv) {

    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Stress testing: -stress-test\n");
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
            "\t\tSet threshold for occurrence of words. Those that appear with "
            "higher frequency in the training data\n");
        printf(
            "\t\twill be randomly down-sampled; default is 1e-3, useful range "
            "is (0, 1e-5)\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
        printf("\t-negative <int>\n");
        printf(
            "\t\tNumber of negative examples; default is 5, common values are "
            "3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf(
            "\t\tThis will discard words that appear less than <int> times; "
            "default is 5\n");
        printf("\t-alpha <float>\n");
        printf(
            "\t\tSet the starting learning rate; default is 0.025 for "
            "skip-gram and 0.05 for CBOW\n");
        printf("\t-classes <int>\n");
        printf(
            "\t\tOutput word classes rather than word vectors; default number "
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
            "\t\tThe vocabulary will be read from <file>, not constructed from "
            "the training data\n");
        printf("\t-cbow <int>\n");
        printf(
            "\t\tUse the continuous bag of words model; default is 1 (use 0 "
            "for skip-gram model)\n");
        printf("\nExamples:\n");
        printf(
            "./word2vec -train data.txt -output vec.txt -size 200 -window 5 "
            "-sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }

    if ((i = ArgPos((char *)"-stress-test", argc, argv)) > 0) {
        assert(i == 1);
        test(argc, argv);
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
    TrainModel();
    return 0;
}
