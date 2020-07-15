#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <igraph.h>
#include <assert.h>
using real = float;
static const long long MAX_VOCAB_SIZE = 10000000;
static const long long MAX_WORDLEN = 200;
long long VOCABSIZE, DIMSIZE;
real *vecs[MAX_VOCAB_SIZE];
char words[MAX_VOCAB_SIZE][MAX_WORDLEN];
static const long long TOPK = 5;
static const real INFTY = 1e9;


using namespace std;
int main(int argc, char **argv) {
    // fast IO please.
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    if (argc != 3) {
        cerr << "usage: " << argv[0] << "<model-path> <to-read>\n";
        return 1;
    }
    FILE *f = fopen(argv[1], "r");
    const long TOREAD = atoll(argv[2]);

    if (!f) {
        cerr << "unable to open file: |"  << argv[1] << "|\n";
        return 1;
    }
    fscanf(f, "%lld %lld", &VOCABSIZE, &DIMSIZE);
    cerr << "VOCABSIZE: " << VOCABSIZE << " | DIMSIZE: " << DIMSIZE << "\n";
    VOCABSIZE = min<long long>(VOCABSIZE, TOREAD);
    cerr << "VOCABSIZE adjusted to: max(" << VOCABSIZE << ", " << TOREAD << ") = |" << VOCABSIZE << "|\n";
    for(int i = 0; i < VOCABSIZE; ++i) {
        fscanf(f, "%s", words[i]);
        cerr << "words[" << i << "] = |" << words[i] << "|";
        // this is filthy, why the fuck is float stored as text? we lose
        // precision. 
        vecs[i] = new real[DIMSIZE];
        for(int j = 0; j < DIMSIZE; ++j) { fscanf(f, "%f", &vecs[i][j]); }
        // debug print vectors so we have a sense that we're loading reasonable data.
        const int DEBUG_PRINT_SIZE = min<int>(5, DIMSIZE);
        for(int j = 0; j < DEBUG_PRINT_SIZE; ++j) { cerr << vecs[i][j] << " "; }
        cerr << "\n";
    }

    // make the graph
    // https://igraph.org/c/doc/igraph-Tutorial.html
    igraph_t g;
    const bool DIRECTED = true;
    cerr << "creating graph...";
    igraph_empty(&g, VOCABSIZE, DIRECTED);

    int *topk_wixs = new int[VOCABSIZE * TOPK];
    real *topk_bestds = new real[VOCABSIZE * TOPK];
    assert(topk_wixs != nullptr);
    assert(topk_bestds != nullptr);

    // initialize
    for(int i = 0; i < VOCABSIZE*TOPK; ++i) { topk_wixs[i] = -INFTY; topk_bestds[i] = -1; }

    // #pragma omp parallel for
    for(int w = 0; w < VOCABSIZE; ++w) { // w for word
        cerr << "w = " << w << " | " << words[w] << "\n";
        real *bestd = topk_bestds + TOPK*w; // best distance, sorted highest to lowest.
        int *wixs = topk_wixs + TOPK*w; // indexes of the words

        // find top-k words.
        for(int o = 0; o < VOCABSIZE; ++o) { // o for other word
            real dot = 0;
            // #pragma omp parallel for reduction(+: dot)
            for(int i = 0; i < DIMSIZE; ++i){ dot += vecs[w][i] * vecs[o][i]; }

            // find location of this word in our collection of words.
            for(int i = 0; i < TOPK; i++) {
                if (bestd[i] > dot) continue;
                // we are better than this index.
                // Copy values forward. So n-2 -> n-1; n-3 -> n-2; ... ; i -> i+1.
                // this creates space at i.
                for(int j = TOPK - 2; j >= i; --j) {
                    bestd[j+1] = bestd[j]; wixs[j+1] = wixs[j]; 
                }

                bestd[i] = dot; wixs[i] = o;
                break;
            }
        }

    }

    // create graph
    cerr << "creating graph...";
    
    // https://igraph.org/c/doc/igraph-Basic.html#igraph_add_edges
    igraph_vector_t es; igraph_vector_init(&es, VOCABSIZE*TOPK*2);
    for(int w = 0; w < VOCABSIZE; ++w) {
        cerr << "w = " << w << " | " << words[w] << "\n";
        real *bestd = topk_bestds + TOPK*w; // best distance, sorted highest to lowest.
        int *wixs = topk_wixs + TOPK*w; // indexes of the words
        for(int i = 0; i < TOPK; ++i) {
            assert(wixs[i] >= 0);
            cerr << "\t- " << words[wixs[i]] << " " << bestd[i] << "\n";
            VECTOR(es)[w*TOPK + 2*i + 0] = w;
            VECTOR(es)[w*TOPK + 2*i + 1] = wixs[i];
            // igraph_add_edge(&g, w, wixs[i]);
        }
    }
    igraph_add_edges(&g, &es, /*attr=*/0);

    cerr << "created graph.\n";
    igraph_integer_t nclusters;
    igraph_vector_t membership, csize;
    igraph_vector_init(&membership, 1);
    igraph_vector_init(&csize, 1);
    igraph_clusters(&g, &membership, &csize, &nclusters, IGRAPH_STRONG);

    cerr << "***number of SCCs: "  << nclusters << "***\n";
    cout << "***Size 2 SCCs:***\n";
    // https://igraph.org/c/doc/igraph-Structural.html#igraph_clusters
    for(int c = 0; c < nclusters; ++c) { // c for closure
        const int sz = VECTOR(csize)[c];
        // cout << "size(SCC[" << c << "]) = " << sz;
        if (sz == 2) {
            cout << "- |";
            for(int v = 0; v < VOCABSIZE; ++v) { // v for vertex.
                if (VECTOR(membership)[v] != c) { continue; }
                cout << words[v] << " ";
            }
            cout << "|\n";
        }
    }
    cerr << "destroyed graph.\n";
    igraph_destroy(&g);

    return 0;
}
