#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <igraph.h>
#include <assert.h>
using namespace std;

using real = float;
static const long long MAX_VOCAB_SIZE = 10000000;
static const long long MAX_WORDLEN = 200;
long long VOCABSIZE, DIMSIZE;
real *vecs[MAX_VOCAB_SIZE];
char words[MAX_VOCAB_SIZE][MAX_WORDLEN];
static const long long TOPK = 50;
static const real INFTY = 1e9;

void check_tri_inequality(int *topk_wixs, real *topk_bestds) {

    cout << "\n====Tris:===\n\n";
    for(int u = 0; u < VOCABSIZE; ++u) {
        for(int i = 0; i < TOPK; ++i) {
            const int v = topk_wixs[TOPK*u+i];
            if (v == u) continue; 
            const real uv = topk_bestds[u*TOPK+i];

            for(int j = 0; j < TOPK; ++j) {
                const int w = topk_wixs[TOPK*v+j];
                if (w == v || w == u) continue;
                const real vw = topk_bestds[v*TOPK+j];

                for(int k = 0; k < TOPK; ++k) {
                    const int z = topk_wixs[TOPK*w+k];
                    if (z != u) continue;

                    const real wu = topk_bestds[w*TOPK+k];
                    // check triangle inequality.
                    if ((uv + vw < wu) ||
                            (vw + wu < uv) ||
                            (wu + uv < vw)) {
                        cout << "FAILURE " 
                            << "u:|" << words[u] << "|  " 
                            << "v:|" << words[v] << "|  " 
                            << "w:|" << words[w] << "| "
                            << "uv:|" << uv << "| " 
                            << "vw:|" << vw << "| " 
                            << "wu:|" << wu << "| " 
                            << "\n" << std::flush;
                        // assert(0 && "triangle inequality failed");
                    }
                }
            }
        }
    }

    cout << "\n======\n";
}

void build_igraph(igraph_t &g, int *topk_wixs, real *topk_bestds) {
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
    igraph_add_edges(&g, &es, nullptr);

    // edges.
    // for(int w = 0; w < VOCABSIZE; ++w) {
    //     for(int i = 0; i < TOPK; ++i) {
    //         assert(wixs[i] >= 0);
    //         igraph_get_eid
    //     }
    // }

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
}


int main(int argc, char **argv) {
    // fast IO please.
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    if (argc != 3) {
        cerr << "usage: " << argv[0] << "<model-path> <num-words-to-read>\n";
        return 1;
    }
    FILE *f = fopen(argv[1], "r");
    const long TOREAD = atoll(argv[2]);

    if (!f) {
        cerr << "unable to open model file: |"  << argv[1] << "|\n";
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

    // normalize the vectors
    printf("normalizing vectors using L2 norm...\n");
    for(int i = 0; i < VOCABSIZE; ++i) {
        real len = 0;
        for (int j = 0; j < DIMSIZE; ++j) { len += vecs[i][j] * vecs[i][j]; }
        len = sqrt(len);
        for (int j = 0; j < DIMSIZE; j++) { vecs[i][j] /= len; }
    }

    printf("exponentiating vectors...\n");

    for(int i = 0; i < VOCABSIZE; ++i) {
        for (int j = 0; j < DIMSIZE; ++j) { vecs[i][j] = exp(vecs[i][j]); }
    }

    printf("normalizing vectors using L1 norm...\n");
    for(int j = 0; j < DIMSIZE; ++j) {
        real total = 0;
        for(int i = 0; i < VOCABSIZE; ++i) { total += vecs[i][j]; }
        for(int i = 0; i < VOCABSIZE; ++i) { vecs[i][j] /= total; }
    }

    printf("discretize vectors using mean...\n");
    for(int j = 0; j < DIMSIZE; ++j) {
        real total = 0;
        for(int i = 0; i < VOCABSIZE; ++i) { total += vecs[i][j]; }
        // TODO: this will just be 1.0 / VOCABSIZE, no?
        const real mean = total / VOCABSIZE;
        for(int i = 0; i < VOCABSIZE; ++i) { 
            vecs[i][j] = vecs[i][j] > mean ? 1 : 0;
        }
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

    #pragma omp parallel for
    for(int w = 0; w < VOCABSIZE; ++w) { // w for word
        // cerr << "w = " << w << " | " << words[w] << "\n";
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

    build_igraph(g, topk_wixs, topk_bestds);
    check_tri_inequality(topk_wixs, topk_bestds);


    return 0;
}
