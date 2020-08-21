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
#include <cooperative_groups.h>
#include <string>
#include <vector>
#include <unordered_map>
using namespace std;
using real = float;


#define GPU_ERRCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      assert(false && "GPU error!");
      // if (abort) { exit(code); };
   }
}


bool is_stop(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '.' || c == ',';
}


__global__ void gpu_init_arrays(const int VOCABSIZE,
        const int DIMSIZE,
        real *pos,
        real *neg) {
    const unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= VOCABSIZE) return;
    real *vpos = pos + x*DIMSIZE;
    real *vneg = neg + x*DIMSIZE;
    unsigned long long next_random = 2 + x;
    for(int i = 0; i < DIMSIZE; ++i) { 
        const float rand01 = (float)(next_random & 0xFFFF)/(65536.0);
        vpos[i] = (rand01 - 0.5) / DIMSIZE;
        next_random = next_random * (unsigned long long)25214903917 + 11;
    }
    for(int i = 0; i < DIMSIZE; ++i) { 
        const float rand01 = (float)(next_random & 0xFFFF)/(65536.0);
        vneg[i] = (rand01 - 0.5) / DIMSIZE;
        next_random = next_random * (unsigned long long)25214903917 + 11;
    }
}


__global__ void gpu_compute_loss(
        const long long PAIRS_PER_BATCH,
        const int DIMSIZE, 
        const long long *w1_ixs,
        const long long *w2_ixs,
        const real *dot_targets,
        const real *pos,
        const real *neg,
        real *loss) {
    const unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= PAIRS_PER_BATCH) return;
    real dot = 0;
    const real *vpos = pos + w1_ixs[x]*DIMSIZE;
    const real *vneg = neg + w2_ixs[x]*DIMSIZE;
    for(int i = 0; i < DIMSIZE; ++i) { dot += vpos[i]*vneg[i]; }
    loss[x] = dot_targets[x] - dot;
}

const float ALPHA = 0.01;

__global__ void gpu_backprop(
        const long long PAIRS_PER_BATCH,
        const int DIMSIZE, 
        const long long *w1_ixs,
        const long long *w2_ixs,
        const real *dot_targets,
        real *pos,
        real *neg,
        const real *loss) {
    const unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= PAIRS_PER_BATCH) return;
    real *vpos = pos + w1_ixs[x]*DIMSIZE;
    real *vneg = neg + w2_ixs[x]*DIMSIZE;
    for(int i = 0; i < DIMSIZE; ++i) { vneg[i] -= ALPHA * loss[x] * vpos[i]; }
    for(int i = 0; i < DIMSIZE; ++i) { vpos[i] -= ALPHA * loss[x] * vneg[i]; }
}




const char *SPINNERS[] = { "|", "/", "-", "\\" };

static const long long MAX_WORDLEN = 512;
static const long long MINFREQ = 5;
static const long long WINDOWSIZE = 8;
static const long long NUM_NEGSAMPLES = 1;
static const long long DIMSIZE = 1;
static const long long BATCHSIZE = 10000;
static const long long NEPOCH = 3;
// word to frequency
unordered_map<string, long long> w2f;
unordered_map<string, long long> w2ix;
static const long long MAX_NUM_WORDS = 1000000000;
vector<string> ix2w;
// corpus as word indexes.
vector<long long> corpus;


int main() {
    printf("min freq: %4lld\n", MINFREQ);
    printf("number of negsamples: %4lld\n", NUM_NEGSAMPLES);
    printf("window size: %4lld\n", WINDOWSIZE);
    printf("dimension size: %4lld\n", DIMSIZE);
    printf("batch size: %4lld\n", BATCHSIZE);

    srand(0);
    FILE *f = fopen("../../utilities/text8", "rb");
    while(!feof(f)) {
        char word[MAX_WORDLEN];

        // eat whitespace
        while(1) {
            int c = fgetc(f);
            if (feof(f)) { break; }
            if (is_stop(c)) { continue; }
            ungetc(c, f);
            break;
        }

        if (feof(f)) { break; }

        int wlen = 0;
        while(1) {
            int c = fgetc(f);
            assert(wlen < MAX_NUM_WORDS);
            if (feof(f)) { break; }
            if (is_stop(c)) { word[wlen++] = '\0'; break; }
            word[wlen++] = c;
        }
        w2f[string(word)]++;

    }
    fclose(f);

    // print words in corpus.
    static const bool DBG_PRINT_ALL_WORDS_IN_CORPUS = false;
    if(DBG_PRINT_ALL_WORDS_IN_CORPUS) {
        for(auto it : w2f) { printf("|%20s|:%4lld\n", it.first.c_str(), it.second); }
    }

    // prune words.
    int wix = 0;
    for(auto it : w2f) {
        if (it.second < MINFREQ) continue;
        ix2w.push_back(it.first);
        w2ix[it.first] = wix;
        wix++;
    }

    static const bool DBG_PRINT_FILTERED_WORDS_IN_CORPUS = false;
    if (DBG_PRINT_FILTERED_WORDS_IN_CORPUS) {
        for(long long i = 0; i < (long long)ix2w.size(); ++i) {
            printf("|%20s|\n", ix2w[i].c_str());
        }
    }


    static const long long VOCABSIZE = ix2w.size();
    printf("vocabulary size: %lld\n", VOCABSIZE);

    // create corpus in corpus[] as a list of ints
    f = fopen("../../utilities/text8", "r");
    while(!feof(f)) {
        char word[MAX_WORDLEN];

        // eat whitespace
        while(1) {
            int c = fgetc(f);
            if (feof(f)) { break; }
            if (is_stop(c)) { continue; }
            ungetc(c, f);
            break;
        }

        if (feof(f)) { break; }

        int wlen = 0;
        while(1) {
            int c = fgetc(f);
            assert(wlen < MAX_NUM_WORDS);
            if (feof(f)) { break; }
            if (is_stop(c)) { word[wlen++] = '\0'; break; }
            word[wlen++] = c;
        }
        corpus.push_back(w2ix[word]);
    }
    fclose(f);
    printf("corpus length: %lld\n", (long long)corpus.size());

    const bool DEBUG_PRINT_SAMPLE_CORPUS = false;
    if (DEBUG_PRINT_SAMPLE_CORPUS) {
        for(int i = 0; i< 1000; ++i) { printf("%s ", ix2w[corpus[i]].c_str()); }
    }


    real *dev_pos, *dev_gpos, *dev_neg, *dev_gneg;
    GPU_ERRCHECK(cudaMalloc((void **)&dev_pos, (long long) VOCABSIZE * DIMSIZE * sizeof(real))); 
    GPU_ERRCHECK(cudaMalloc((void **)&dev_gpos, (long long) VOCABSIZE * DIMSIZE * sizeof(real)));
    GPU_ERRCHECK(cudaMalloc((void **)&dev_neg, (long long) VOCABSIZE * DIMSIZE * sizeof(real)));
    GPU_ERRCHECK(cudaMalloc((void **)&dev_gneg, (long long) VOCABSIZE * DIMSIZE * sizeof(real)));

    static const int PAIRS_PER_BATCH = (2*WINDOWSIZE*(1+NUM_NEGSAMPLES))*BATCHSIZE;
    long long *dev_w1_ixs, *dev_w2_ixs;
    real *dev_dot_targets, *dev_loss;
    GPU_ERRCHECK(cudaMalloc((void **)&dev_w1_ixs, (long long) PAIRS_PER_BATCH * sizeof(long long)));
    GPU_ERRCHECK(cudaMalloc((void **)&dev_w2_ixs, (long long) PAIRS_PER_BATCH * sizeof(long long )));
    GPU_ERRCHECK(cudaMalloc((void **)&dev_dot_targets, (long long) PAIRS_PER_BATCH * sizeof(real)));
    GPU_ERRCHECK(cudaMalloc((void **)&dev_loss, (long long) PAIRS_PER_BATCH  * sizeof(real)));

    gpu_init_arrays<<<(1 + (VOCABSIZE / 1024)), (1024)>>>(VOCABSIZE, DIMSIZE, dev_pos, dev_neg);

    long long *w1_ixs = (long long*)malloc(PAIRS_PER_BATCH * sizeof(long));
    long long *w2_ixs = (long long*)malloc(PAIRS_PER_BATCH * sizeof(long));
    real *dot_targets = (real*)malloc(PAIRS_PER_BATCH * sizeof(real));

    for(long long e = 0; e < NEPOCH; ++e) {
        printf("\n===epoch %4lld/%4lld===\n", e+1, NEPOCH);
        int count = 0;
        for(long long f = WINDOWSIZE; f < (long long)corpus.size() - WINDOWSIZE; ++f) {
            if (f % 100 == 0) { 
                printf("\r%4lld/%4lld %4.2f", 
                    f, (long long)corpus.size(), (100.0*f)/corpus.size());
            }
            for(long long w = -WINDOWSIZE; w <= WINDOWSIZE; ++w) {
                if (w == 0) { continue; }
                // if (f + w < 0 || f + w >= (long long)corpus.size()) continue;
                w1_ixs[count] = corpus[f];
                w2_ixs[count] = corpus[f+w];
                dot_targets[count] = 0;
                count++;
                assert(count <= PAIRS_PER_BATCH && "focus word");
                for(int r = 0; r < NUM_NEGSAMPLES; ++r) {
                    w1_ixs[count] = f;
                    w2_ixs[count] = rand() % VOCABSIZE;
                    dot_targets[count] = 1;
                    count++;
                }
                assert(count <= PAIRS_PER_BATCH && "negsampling");
            }
            // we haven't done enough vectors yet
            if (count != PAIRS_PER_BATCH) { continue; }
            // we need to run some vectors
            count = 0;
            printf("\rprocessing...                                     ");
            cudaMemcpy(dev_dot_targets, dot_targets, PAIRS_PER_BATCH*sizeof(real), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_w1_ixs, w1_ixs, PAIRS_PER_BATCH*sizeof(long long), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_w2_ixs, w2_ixs, PAIRS_PER_BATCH*sizeof(long long), cudaMemcpyHostToDevice);
            gpu_compute_loss<<<(1 + (PAIRS_PER_BATCH / 1024)), (1024)>>>(PAIRS_PER_BATCH,
                    DIMSIZE,
                    dev_w1_ixs,
                    dev_w2_ixs,
                    dev_dot_targets,
                    dev_pos,
                    dev_neg,
                    dev_loss);
            gpu_backprop<<<(1 + (PAIRS_PER_BATCH/1024)), 1024>>>(PAIRS_PER_BATCH,
                    DIMSIZE,
                    dev_w1_ixs,
                    dev_w2_ixs,
                    dev_dot_targets,
                    dev_pos,
                    dev_neg,
                    dev_loss);
        }
    }

    real *pos = (real*)malloc(VOCABSIZE * DIMSIZE*sizeof(real));
    cudaMemcpy(pos, dev_pos, VOCABSIZE*DIMSIZE*sizeof(real), cudaMemcpyDeviceToHost); 

    const char *POS_WRITE_FILEPATH = "gpuout.bin";
    printf("\nwriting output to |%s|\n", POS_WRITE_FILEPATH);
    FILE *fo = fopen(POS_WRITE_FILEPATH, "wb");
    fprintf(fo, "%lld %lld\n", VOCABSIZE, DIMSIZE);

    for (int i = 0; i < VOCABSIZE; ++i) {
        fprintf(fo, "%s ", ix2w[i].c_str());
        for (int j = 0; j < DIMSIZE; ++j) {
            // pos[i*DIMSIZE+j] = ((rand() % 100) / 100.0);
            fwrite(&pos[i*DIMSIZE+j], sizeof(real), 1, fo);
        }
        fprintf(fo, "\n");
    }
    fclose(fo);

    return 0;
}
