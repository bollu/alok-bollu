#include <map>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <gurobi_c++.h>
#include <assert.h>
using namespace std;
using real = float;

bool is_stop(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '.' || c == ',';
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


int main(int argc, char **argv) {
    if (argc != 2) { 
        printf("usage: %s <path-to-corupus>\n", argv[0]); return 1;
    }
    printf("min freq: %4lld\n", MINFREQ);
    printf("number of negsamples: %4lld\n", NUM_NEGSAMPLES);
    printf("window size: %4lld\n", WINDOWSIZE);
    printf("dimension size: %4lld\n", DIMSIZE);
    printf("batch size: %4lld\n", BATCHSIZE);

    srand(0);
    FILE *f = fopen(argv[1], "rb");
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
    f = fopen(argv[1], "r");
    if (f == NULL) { printf("ERROR: unable to open file: |%s|\n", argv[1]); return 1; }
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

    // zero by default
    real *dots_targets = (real *)calloc(VOCABSIZE * VOCABSIZE, sizeof(real));

    printf("building co-occurence matrix...\n");
    for(long long f = WINDOWSIZE; f < (long long)corpus.size() - WINDOWSIZE; ++f) {
        const int fwix = corpus[f];
        if (f % 100 == 0) { 
            printf("\r%4lld/%4lld %4.2f", 
                    f, (long long)corpus.size(), (100.0*f)/corpus.size());
        }
        for(long long offset = -WINDOWSIZE; offset <= WINDOWSIZE; ++offset) {
            const int cwix = corpus[f+offset];
            dots_targets[fwix*VOCABSIZE+cwix]++;
            dots_targets[cwix*VOCABSIZE+fwix]++;
        }
    }

    printf("training\n");
    GRBEnv grb_env;
    GRBModel grb_model(grb_env);
    real *pos = (real*)malloc(VOCABSIZE * DIMSIZE*sizeof(real));
    for(long long e = 0; e < NEPOCH; ++e) {
        printf("\r");
        printf("[%10lld/%10lld]", e, NEPOCH);


        if (e % 100 != 0) continue;
        const char *POS_WRITE_FILEPATH = "glove-optimized.bin";
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
        printf("[wrote |%20s|]\n", POS_WRITE_FILEPATH);
    }

    return 0;
}

