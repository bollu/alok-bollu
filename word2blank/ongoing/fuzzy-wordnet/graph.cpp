#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <igraph.h>
#include <assert.h>
#include <vector>
using namespace std;

using real = float;
static const long long MAX_VOCAB_SIZE = 1000000;
static const long long MAX_WORDLEN = 200;
long long VOCABSIZE, DIMSIZE;
real *vecs[MAX_VOCAB_SIZE];
char words[MAX_VOCAB_SIZE][MAX_WORDLEN];
vector<long> adj_list[MAX_VOCAB_SIZE];
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
    cerr << "TOREAD: " << TOREAD << " | MAX_VOCAB_SIZE: " << MAX_VOCAB_SIZE << "\n";
    VOCABSIZE = min<long long>(VOCABSIZE, TOREAD);
    cerr << "VOCABSIZE adjusted to: max(" << VOCABSIZE << ", " << TOREAD << ") = |" << VOCABSIZE << "|\n";
    assert(VOCABSIZE < MAX_VOCAB_SIZE);

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

    /*
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
    */


    real *dots = new real[VOCABSIZE*VOCABSIZE];

#pragma omp parallel for
    for(long long i = 0; i < VOCABSIZE*VOCABSIZE; ++i) {
        const long long w1 = i/VOCABSIZE;
        const long long w2 = i%VOCABSIZE;
        // #pragma omp parallel for reduction(+: dot)
        float dot = 0;
        for(int i = 0; i < DIMSIZE; ++i){ dot += vecs[w1][i] * vecs[w2][i]; }

        assert(fabs(dot) < 2);
        dots[w1*VOCABSIZE + w2] = dots[w2*VOCABSIZE+w1] = max<double>(0, 1.3 - fabs(dot));
        assert(dots[w1*VOCABSIZE+w2] >= 0);
        assert(dots[w2*VOCABSIZE+w1] >= 0);
    }

fprintf(stderr, "computing adjacency list...\n");

float cutoff = 0;
// compute pruned adjacency list.
for(int i = 0; i < VOCABSIZE; ++i) {
    for(int j = i+1; j < VOCABSIZE; ++j) {
        if (dots[i*VOCABSIZE+j] >= cutoff) {
            adj_list[i].push_back(j);
            adj_list[j].push_back(i);
        }
    }
    if (adj_list[i].size() > 0) {
        printf("#ws adj to |%20s|: %lld\n",
                words[i],
                (long long)adj_list[i].size());
    }
}

fprintf(stderr, "Cutoff: %4.2f\n", cutoff);
fprintf(stderr, "Done.\n");
fprintf(stderr, "Press-key>.\n"); getchar();

fprintf(stderr, "\nProgress...");
for(long long w1 = 0; w1 < VOCABSIZE; ++w1) {
    for(long long ixw2 = 0; ixw2 < adj_list[w1].size(); ++ixw2) {
        const long long w2 = adj_list[w1][ixw2];
        for(long long ixw3 = 0; ixw3 < adj_list[w2].size(); ++ixw3) {
            const long long w3 = adj_list[w2][ixw3];
            if (w1 == w2) continue;
            if (w1 == w3) continue;
            if (w2 == w3) continue;
            // (w1, w2) < (w1, w3) + (w3, w2)
            const real l12 = dots[w1*VOCABSIZE+w2];
            const real l23 = dots[w2*VOCABSIZE+w3];
            const real l31 = dots[w1*VOCABSIZE+w3];
            if (l12 < (l23 + l31)) { 
                continue;

                // fprintf(stdout,
                //         "\n|%s|--%4.2f--|%s|--%4.2f--|%s|--%4.2f--|%s|: %4.2f < %4.2f + %4.2f = %4.2f\n",
                //         words[w1], l12, words[w2], l23, words[w3], l31, words[w1],
                //         l12, l23, l31, l23 + l31);
            }
            // if (l12 - (l23 + l31) < 1e-2) { continue; }
            fprintf(stdout,
                    "\n|%s|--%4.2f--|%s|--%4.2f--|%s|--%4.2f--|%s|: %4.2f !< %4.2f + %4.2f = %4.2f\n",
                    words[w1], l12, words[w2], l23, words[w3], l31, words[w1],
                    l12, l23, l31, l23 + l31);
            // assert(false);
        }
    }
}

    return 0;
}
