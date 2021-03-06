//  GloVe: Global Vectors for Word Representation
//
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
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
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

// silence the many complaints from visual studio
#define _CRT_SECURE_NO_WARNINGS

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cassert>
#include <time.h>
#include <armadillo>
// windows pthread.h is buggy, but this #define fixes it
#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>
#include "grad.h"
#include "common.h"

using namespace std;
#define _FILE_OFFSET_BITS 64
#define STRERROR(ERRNO, BUF, BUFSIZE) strerror_r((ERRNO), (BUF), (BUFSIZE))

int write_header=0; //0=no, 1=yes; writes vocab_size/vector_size as first line for use with some libraries, such as gensim.
int verbose = 2; // 0, 1, or 2
int seed = 0;
int use_unk_vec = 0; // 0 or 1
long long int P = -1;
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
long long int vector_size = 50; // Word vector size
int save_gradsq = 0; // By default don't save squared gradient values
int use_binary = 0; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
int checkpoint_every = 0; // checkpoint the model for every checkpoint_every iterations. Do nothing if checkpoint_every <= 0
int load_init_param = 0; // if 1 initial paramters are loaded from -init-param-file
int save_init_param = 0; // if 1 initial paramters are saved (i.e., in the 0 checkpoint)
int load_init_gradsq = 0; // if 1 initial squared gradients are loaded from -init-gradsq-file
realglove eta = 0.05; // Initial learning rate
realglove alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
realglove grad_clip_value = 100.0; // Clipping parameter for gradient components. Values will be clipped to [-grad_clip_value, grad_clip_value] interval.
realglove *W, *gradsq, *cost;
//syn0_mat stores global matrix representation of each word
arma::cube syn0;
//syn1neg_mat stores context matrix representation of each word
arma::cube syn1neg;
//grad_mat stores gradient of each matrix
arma::cube syn0_gradsq;
arma::cube syn1neg_gradsq;
long long num_lines, *lines_per_thread, vocab_size;
char vocab_file[MAX_STRING_LENGTH];
char input_file[MAX_STRING_LENGTH];
char save_W_file[MAX_STRING_LENGTH];
char save_gradsq_file[MAX_STRING_LENGTH];
char init_param_file[MAX_STRING_LENGTH];
char init_gradsq_file[MAX_STRING_LENGTH];

int g_scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return (*s1 - *s2);
}

int arg_pos(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if (!g_scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int log_file_loading_err(const char*file_description, char *file_name) {
    fprintf(stderr, "Unable to open %s %s.\n", file_description, file_name);
    fprintf(stderr, "Errno: %d\n", errno);
    char error[MAX_STRING_LENGTH];
    STRERROR(errno, error, MAX_STRING_LENGTH);
    fprintf(stderr, "Error description: %s\n", error);
    return errno;
}


/**
 * Loads a save file for use as the initial values for the parameters or gradsq
 * Return value: 0 if success, -1 if fail
 */
int load_init_file(char* file_name, realglove *array, long long array_size) {
    FILE *fin;
    long long a;
    fin = fopen(file_name, "rb");
    if (fin == NULL) {
        log_file_loading_err("init file", file_name);
        return -1;
    }
    for (a = 0; a < array_size; a++) {
        if (feof(fin)) {
            fprintf(stderr, "EOF reached before data fully loaded in %s.\n", file_name);
            fclose(fin);
            return -1;
        }
        fread(&array[a], sizeof(realglove), 1, fin);
    }
    fclose(fin);
    return 0;
}

void initialize_parameters() {
    // TODO: return an error code when an error occurs, clean up in the calling routine
    if (seed == 0) {
        seed = time(0);
    }
    fprintf(stderr, "Using random seed %d\n", seed);
    srand(seed);
    long long a;
    long long W_size = 2 * vocab_size * (vector_size + 1); // +1 to allocate space for bias

    /* Allocate space for word vectors and context word vectors, and corresponding gradsq */
    //keeping it uncommented for functions which are using it but we do not require    
    a = posix_memalign((void **)&W, 128, W_size * sizeof(realglove)); // Might perform better than malloc
    if (W == NULL) {
        fprintf(stderr, "Error allocating memory for W\n");
        exit(1);
    }
    a = posix_memalign((void **)&gradsq, 128, W_size * sizeof(realglove)); // Might perform better than malloc
    if (gradsq == NULL) {
        fprintf(stderr, "Error allocating memory for gradsq\n");
        free(W);
        exit(1);
    }

    //Allocate space to syn0 cube(no bias)
    syn0.set_size(vector_size, P, vocab_size);
    printf("syn0.n_slices: %lld\n", syn0.n_slices);
    assert((long long)syn0.n_slices == vocab_size);
    
    //Allocate space to syn1neg cube
    syn1neg.set_size(vector_size, P, vocab_size);
    printf("syn1neg.n_slices:%lld\n", syn1neg.n_slices);
    assert((long long)syn1neg.n_slices == vocab_size);

    //Allocate space to syn0_grad cube
    syn0_gradsq.set_size(vector_size, P, vocab_size);
    printf("syn0_gradsq.n_slices: %lld\n", syn0_gradsq.n_slices);
    assert((long long)syn0_gradsq.n_slices == vocab_size);

    //Allocate space to syn1neg_grad cube
    syn1neg_gradsq.set_size(vector_size, P, vocab_size);
    printf("syn1neg_gradsq.n_slices:%lld\n", syn1neg_gradsq.n_slices);
    assert((long long)syn1neg_gradsq.n_slices == vocab_size);


    if (load_init_param) {
        // Load existing parameters
        fprintf(stderr, "\nLoading initial parameters from %s \n", init_param_file);
        if (load_init_file(init_param_file, W, W_size)) {
            free(W);
            free(gradsq);
            exit(1);
        }
    } else {
        // Initialize new parameters for original code word vectors(keeping it uncommented)
        for (a = 0; a < W_size; ++a) {
            W[a] = (rand() / (realglove)RAND_MAX - 0.5) / vector_size;
        }
        //Initalising the matrices naively
        //Have to check "Statistics on Special Manifold" for a smarter method
        for (a = 0; a < vocab_size; a++) {
            printf("\rinitializing Syn0 |%lld|", a);
            arma::mat X = arma::orth(arma::randn<arma::mat>(vector_size, P));
            arma::uword r = arma::rank(X);
            if ((long long)r == P) syn0.slice(a) = X;
            else printf("FULL COLUMN FAIL\n");
        }
        for (a = 0; a < vocab_size; a++) {
            printf("\rinitializing Syn1neg |%lld|", a);
            arma::mat X = arma::orth(arma::randn<arma::mat>(vector_size, P));
            arma::uword r = arma::rank(X);
            if ((long long)r == P) syn1neg.slice(a) = X;
            else printf("FULL COLUMN FAIL\n");
        }

    }

    if (load_init_gradsq) {
        // Load existing squared gradients
        fprintf(stderr, "\nLoading initial squared gradients from %s \n", init_gradsq_file);
        if (load_init_file(init_gradsq_file, gradsq, W_size)) {
            free(W);
            free(gradsq);
            exit(1);
        }
    } else {
        // Initialize new squared gradients of original code(keeping it uncommented)
        for (a = 0; a < W_size; ++a) {
            gradsq[a] = 1.0; // So initial value of eta is equal to initial learning rate
        }

        //Initialize our grad matrix (initilaise with ones? or randomly?)
        // for (a = 0; a < vocab_size; a++) {
        //     printf("\rinitializing syn1neg |%d|", a);
        //     arma::mat X = arma::orth(arma::randn<arma::mat>(vector_size, P));
        //     arma::uword r = arma::rank(X);
        //     if ((long long)r == P) grad_mat.slice(a) = X;
        //     else printf("FULL COLUMN FAIL\n");
        // }
        syn0_gradsq.zeros();
        syn1neg_gradsq.zeros();
    }
}

inline realglove check_nan(realglove update) {
    if (isnan(update) || isinf(update)) {
        fprintf(stderr,"\ncaught NaN in update");
        return 0.;
    } else {
        return update;
    }
}

/* Train the GloVe model */
void *glove_thread(void *vid) {
    long long a, b ,l1, l2;
    long long id = *(long long*)vid;
    CREC cr;
    realglove diff, fdiff, distance;
    realglove clampval = 0.005;
    FILE *fin;
    fin = fopen(input_file, "rb");
    if (fin == NULL) {
        // TODO: exit all the threads or somehow mark that glove failed
        log_file_loading_err("input file", input_file);
        pthread_exit(NULL);
    }
    fseeko(fin, (num_lines / num_threads * id) * (sizeof(CREC)), SEEK_SET); //Threads spaced roughly equally throughout file
    cost[id] = 0;
    
    // realglove* W_updates1 = (realglove*)malloc(vector_size * sizeof(realglove));
    // if (NULL == W_updates1){
    //     fclose(fin);
    //     pthread_exit(NULL);
    // }
    // realglove* W_updates2 = (realglove*)malloc(vector_size * sizeof(realglove));
    //     if (NULL == W_updates2){
    //     fclose(fin);
    //     free(W_updates1);
    //     pthread_exit(NULL);
    // }
    arma::mat syn0_updates(vector_size, P);
    arma::mat syn1neg_updates(vector_size, P);
    arma::mat syn0_metric_grad(vector_size, P);
    arma::mat syn1neg_metric_grad(vector_size, P);
    arma::Mat<double> clamp_mat(vector_size, P); clamp_mat.fill(clampval); 
    for (a = 0; a < lines_per_thread[id]; a++) {
        fread(&cr, sizeof(CREC), 1, fin);
        if (feof(fin)) break;
        if (cr.word1 < 1 || cr.word2 < 1) { continue; }        
        /* Get location of words in W & gradsq */
        //l1 = (cr.word1 - 1LL) * (vector_size + 1); // cr word indices start at 1
        //l2 = ((cr.word2 - 1LL) + vocab_size) * (vector_size + 1); // shift by vocab_size to get separate vectors for context words
        l1 = (cr.word1 - 1LL);
        l2 = (cr.word2 - 1LL);
        /* Calculate cost, save diff for gradients */
        // diff = 0;
        // distance = 0;
        // syn0_metric_grad.zeros();
        // syn1neg_metric_grad.zeros();
        // getDotAndGradients_chordalfrobenius(syn0.slice(l1), syn1neg.slice(l2), distance, syn0_metric_grad, syn1neg_metric_grad); // chordal distance of word and context word matrix
        // //not using bias
        // realglove max_dist = sqrt(P);
	    // diff = (max_dist-distance) - log(cr.val);
        diff = 0;
        distance = 0;
        syn0_metric_grad.zeros();
        syn1neg_metric_grad.zeros();
        getDotAndGradients_chordalfrobenius(syn0.slice(l1), syn1neg.slice(l2), distance, syn0_metric_grad, syn1neg_metric_grad); // chordal distance of word and context word matrix
        //not using bias
        // added sqrt based on: http://helper.ipam.ucla.edu/publications/glws1/glws1_15465.pdf
        const realglove max_dist = sqrt(P);
        // map [0, sqrt(P) -> [1, 0]
        const realglove dot = (max_dist - distance)/max_dist;
        if (dot < 0) {
            printf("\nmax_dist: %4.2f | distance: %4.2f | (max_dist - distance): %4.2f\n", max_dist, distance, max_dist - distance);
        }
        //assert(dot >= 0);
        //assert(dot <= 1);
        fdiff = (cr.val > x_max) ? diff : pow(cr.val / x_max, alpha) * diff; // multiply weighting function (f) with diff
        // Check for NaN and inf() in the diffs.
        if (isnan(diff) || isnan(fdiff) || isinf(diff) || isinf(fdiff)) {
            fprintf(stderr,"Caught NaN in diff for kdiff for thread. Skipping update");
            continue;
        }
        cost[id] += 0.5 * fdiff * diff; // weighted squared error
        /* Adaptive gradient updates */
        realglove syn0_updates_sum = 0;
        realglove syn1neg_updates_sum = 0;
        syn0_updates.zeros();
        syn1neg_updates.zeros();
        arma::Mat<double> temp1 = arma::clamp(-syn0_metric_grad, -grad_clip_value, grad_clip_value)*eta;
        arma::Mat<double> temp2 = arma::clamp(-syn1neg_metric_grad, -grad_clip_value, grad_clip_value)*eta;
        //Calculating grad*eta o 1/sqrt(I + r) for syn0 and syn1neg
        syn0_updates = temp1 / (arma::sqrt(syn0_gradsq.slice(l1)) + clamp_mat);
        syn1neg_updates = temp2 / (arma::sqrt(syn1neg_gradsq.slice(l2)) + clamp_mat);
        syn0_updates_sum = arma::accu(syn0_updates);
        syn1neg_updates_sum = arma::accu(syn1neg_updates);
        //Calculating the matrix r for syn0 and syn1neg which is hadamard product of gradient  
        syn0_gradsq.slice(l1) += temp1%temp1; 
        syn1neg_gradsq.slice(l2) += temp2%temp2;

        if (!isnan(syn0_updates_sum) && !isinf(syn0_updates_sum) && !isnan(syn1neg_updates_sum) && !isinf(syn1neg_updates_sum)) {
            syn0.slice(l1) -= syn0_updates;
            syn1neg.slice(l2) -= syn1neg_updates;
            syn0.slice(l1) = arma::orth(syn0.slice(l1));
            syn1neg.slice(l2) = arma::orth(syn1neg.slice(l2));
        }

        // updates for bias terms
        // W[vector_size + l1] -= check_nan(fdiff / sqrt(gradsq[vector_size + l1]));
        // W[vector_size + l2] -= check_nan(fdiff / sqrt(gradsq[vector_size + l2]));
        // fdiff *= fdiff;
        // gradsq[vector_size + l1] += fdiff;
        // gradsq[vector_size + l2] += fdiff;
        
    }
    // free(W_updates1);
    // free(W_updates2);
    
    fclose(fin);
    pthread_exit(NULL);
}

/* Save params to file */
int save_params(int nb_iter) {
    /*
     * nb_iter is the number of iteration (= a full pass through the cooccurrence matrix).
     *   nb_iter  > 0 => checkpointing the intermediate parameters, so nb_iter is in the filename of output file.
     *   nb_iter == 0 => checkpointing the initial parameters
     *   else         => saving the final paramters, so nb_iter is ignored.
     */

    long long a, b;
    char format[20];
    char output_file[MAX_STRING_LENGTH+20], output_file_gsq[MAX_STRING_LENGTH+20];
    char *word = (char *)malloc(sizeof(char) * MAX_STRING_LENGTH + 1);
    if (NULL == word) {
        return 1;
    }
    FILE *fid, *fout;
    FILE *fgs = NULL;
    
    if (use_binary > 0 || nb_iter == 0) {
        // Save parameters in binary file
        // note: always save initial parameters in binary, as the reading code expects binary
        if (nb_iter < 0)
            sprintf(output_file,"%s.bin",save_W_file);
        else
            sprintf(output_file,"%s.%03d.bin",save_W_file,nb_iter);

        fout = fopen(output_file,"wb");
        if (fout == NULL) {log_file_loading_err("weights file", save_W_file); free(word); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if (fid == NULL) {log_file_loading_err("vocab file", vocab_file); free(word); fclose(fout); return 1;}
        if (write_header) fprintf(fout, "%lld %lld %lld\n", vocab_size, vector_size, P);
        for (a = 0; a < vocab_size; a++) {
            if (fscanf(fid,format,word) == 0) {free(word); fclose(fid); fclose(fout); return 1;}
            // input vocab cannot contain special <unk> keyword
            if (strcmp(word, "<unk>") == 0) {free(word); fclose(fid); fclose(fout);  return 1;}
            fprintf(fout, "%s ",word);
            for(b = 0; b < P; b++) for (long long c = 0; c < vector_size; c++) fwrite(&syn0(c, b, a), sizeof(realglove), 1, fout);
            fprintf(fout, "\n");
            if (fscanf(fid,format,word) == 0) {
            	// Eat irrelevant frequency entry
            	fclose(fout);
            	fclose(fid);
            	free(word);
	     }
        } 
        if (save_gradsq > 0) {
            if (nb_iter < 0)
                sprintf(output_file_gsq,"%s.bin",save_gradsq_file);
            else
                sprintf(output_file_gsq,"%s.%03d.bin",save_gradsq_file,nb_iter);

            fgs = fopen(output_file_gsq,"wb");
            if (fgs == NULL) {log_file_loading_err("gradsq file", save_gradsq_file); free(word); return 1;}
            for (a = 0; a < 2 * vocab_size * (vector_size + 1); a++) fwrite(&gradsq[a], sizeof(realglove), 1,fgs);
            fclose(fgs);
        }
    }
    if (use_binary != 1) { // Save parameters in text file
        if (nb_iter < 0)
            sprintf(output_file,"%s.txt",save_W_file);
        else
            sprintf(output_file,"%s.%03d.txt",save_W_file,nb_iter);
        if (save_gradsq > 0) {
            if (nb_iter < 0)
                sprintf(output_file_gsq,"%s.txt",save_gradsq_file);
            else
                sprintf(output_file_gsq,"%s.%03d.txt",save_gradsq_file,nb_iter);

            fgs = fopen(output_file_gsq,"wb");
            if (fgs == NULL) {log_file_loading_err("gradsq file", save_gradsq_file); free(word); return 1;}
        }
        fout = fopen(output_file,"wb");
        if (fout == NULL) {log_file_loading_err("weights file", save_W_file); free(word); return 1;}
        fid = fopen(vocab_file, "r");
        sprintf(format,"%%%ds",MAX_STRING_LENGTH);
        if (fid == NULL) {log_file_loading_err("vocab file", vocab_file); free(word); fclose(fout); return 1;}
        if (write_header) fprintf(fout, "%lld %lld %lld\n", vocab_size, vector_size, P);
        for (a = 0; a < vocab_size; a++) {
            if (fscanf(fid,format,word) == 0) {free(word); fclose(fid); fclose(fout); return 1;}
            // input vocab cannot contain special <unk> keyword
            if (strcmp(word, "<unk>") == 0) {free(word); fclose(fid); fclose(fout);  return 1;}
            fprintf(fout, "%s",word);
            if (model == 0) { // Save all parameters (including bias)
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b]);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", W[(vocab_size + a) * (vector_size + 1) + b]);
            }
            if (model == 1) // Save only "word" vectors (without bias)
               for(b = 0; b < P; b++) for (long long c = 0; c < vector_size; c++) fprintf(fout," %lf", syn0(c, b, a));
            if (model == 2) // Save "word + context word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", W[a * (vector_size + 1) + b] + W[(vocab_size + a) * (vector_size + 1) + b]);
            fprintf(fout,"\n");
            if (save_gradsq > 0) { // Save gradsq
                fprintf(fgs, "%s",word);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[a * (vector_size + 1) + b]);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fgs," %lf", gradsq[(vocab_size + a) * (vector_size + 1) + b]);
                fprintf(fgs,"\n");
            }
            if (fscanf(fid,format,word) == 0) {
                // Eat irrelevant frequency entry
                fclose(fout);
                fclose(fid);
                free(word); 
                return 1;
                } 
        }

        if (use_unk_vec) {
            realglove* unk_vec = (realglove*)calloc((vector_size + 1), sizeof(realglove));
            realglove* unk_context = (realglove*)calloc((vector_size + 1), sizeof(realglove));
            strcpy(word, "<unk>");

            long long num_rare_words = vocab_size < 100 ? vocab_size : 100;

            for (a = vocab_size - num_rare_words; a < vocab_size; a++) {
                for (b = 0; b < (vector_size + 1); b++) {
                    unk_vec[b] += W[a * (vector_size + 1) + b] / num_rare_words;
                    unk_context[b] += W[(vocab_size + a) * (vector_size + 1) + b] / num_rare_words;
                }
            }

            fprintf(fout, "%s",word);
            if (model == 0) { // Save all parameters (including bias)
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_vec[b]);
                for (b = 0; b < (vector_size + 1); b++) fprintf(fout," %lf", unk_context[b]);
            }
            if (model == 1) // Save only "word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b]);
            if (model == 2) // Save "word + context word" vectors (without bias)
                for (b = 0; b < vector_size; b++) fprintf(fout," %lf", unk_vec[b] + unk_context[b]);
            fprintf(fout,"\n");

            free(unk_vec);
            free(unk_context);
        }

        fclose(fid);
        fclose(fout);
        if (save_gradsq > 0) fclose(fgs);
    }
    free(word);
    return 0;
}

/* Train model */
int train_glove() {
    long long a, file_size;
    int save_params_return_code;
    int b;
    FILE *fin;
    realglove total_cost = 0;

    fprintf(stderr, "TRAINING MODEL\n");
    
    fin = fopen(input_file, "rb");
    if (fin == NULL) {log_file_loading_err("cooccurrence file", input_file); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);
    num_lines = file_size/(sizeof(CREC)); // Assuming the file isn't corrupt and consists only of CREC's
    fclose(fin);
    fprintf(stderr,"Read %lld lines.\n", num_lines);
    if (verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if (verbose > 1) fprintf(stderr,"done.\n");
    if (save_init_param) {
        if (verbose > 1) fprintf(stderr,"Saving initial parameters... ");
        save_params_return_code = save_params(0);
        if (save_params_return_code != 0)
            return save_params_return_code;
        if (verbose > 1) fprintf(stderr,"done.\n");
    }
    if (verbose > 0) fprintf(stderr,"N: %d\n", vector_size);
    if (verbose > 0) fprintf(stderr,"P:%d\n", P);
    if (verbose > 0) fprintf(stderr,"vocab size: %lld\n", vocab_size);
    if (verbose > 0) fprintf(stderr,"x_max: %lf\n", x_max);
    if (verbose > 0) fprintf(stderr,"alpha: %lf\n", alpha);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    lines_per_thread = (long long *) malloc(num_threads * sizeof(long long));
    
    time_t rawtime;
    struct tm *info;
    char time_buffer[80];
    // Lock-free asynchronous SGD
    for (b = 0; b < num_iter; b++) {
        total_cost = 0;
        for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
        lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
        long long *thread_ids = (long long*)malloc(sizeof(long long) * num_threads);
        for (a = 0; a < num_threads; a++) thread_ids[a] = a;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)&thread_ids[a]);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        for (a = 0; a < num_threads; a++) total_cost += cost[a];
        free(thread_ids);

        time(&rawtime);
        info = localtime(&rawtime);
        strftime(time_buffer,80,"%x - %I:%M.%S%p", info);
        fprintf(stderr, "%s, iter: %03d, cost: %lf\n", time_buffer,  b+1, total_cost/num_lines);

        if (checkpoint_every > 0 && (b + 1) % checkpoint_every == 0) {
            fprintf(stderr,"    saving intermediate parameters for iter %03d...", b+1);
            save_params_return_code = save_params(b+1);
            if (save_params_return_code != 0) {
                free(pt);
                free(lines_per_thread);
                return save_params_return_code;
            }
            fprintf(stderr,"done.\n");
        }
    }
    free(pt);
    free(lines_per_thread);
    return save_params(-1);
}

int main(int argc, char **argv) {
    int i;
    FILE *fid;
    int result = 0;
    
    if (argc == 1) {
        printf("GloVe: Global Vectors for Word Representation, v0.2\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-write-header <int>\n");
        printf("\t\tIf 1, write vocab_size/vector_size as first line. Do nothing if 0 (default).\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t\tParameter in exponent of weighting function; default 0.75\n");
        printf("\t-x-max <float>\n");
        printf("\t\tParameter specifying cutoff in weighting function; default 100.0\n");
        printf("\t-grad-clip\n");
        printf("\t\tGradient components clipping parameter. Values will be clipped to [-grad-clip, grad-clip] interval\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave output in binary format (0: text, 1: binary, 2: both); default 0\n");
        printf("\t-model <int>\n");
        printf("\t\tModel for word vector output (for text output only); default 2\n");
        printf("\t\t   0: output all data, for both word and context word vectors, including bias terms\n");
        printf("\t\t   1: output word vectors, excluding bias terms\n");
        printf("\t\t   2: output word vectors + context word vectors, excluding bias terms\n");
        printf("\t-input-file <file>\n");
        printf("\t\tBinary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin\n");
        printf("\t-vocab-file <file>\n");
        printf("\t\tFile containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt\n");
        printf("\t-save-file <file>\n");
        printf("\t\tFilename, excluding extension, for word vector output; default vectors\n");
        printf("\t-gradsq-file <file>\n");
        printf("\t\tFilename, excluding extension, for squared gradient output; default gradsq\n");
        printf("\t-save-gradsq <int>\n");
        printf("\t\tSave accumulated squared gradients; default 0 (off); ignored if gradsq-file is specified\n");
        printf("\t-checkpoint-every <int>\n");
        printf("\t\tCheckpoint a  model every <int> iterations; default 0 (off)\n");
        printf("\t-load-init-param <int>\n");
        printf("\t\tLoad initial parameters from -init-param-file; default 0 (false)\n");
        printf("\t-save-init-param <int>\n");
        printf("\t\tSave initial parameters (i.e., checkpoint the model before any training); default 0 (false)\n");
        printf("\t-init-param-file <file>\n");
        printf("\t\tBinary initial parameters file to be loaded if -load-init-params is 1; (default is to look for vectors.000.bin)\n");
        printf("\t-load-init-gradsq <int>\n");
        printf("\t\tLoad initial squared gradients from -init-gradsq-file; default 0 (false)\n");
        printf("\t-init-gradsq-file <file>\n");
        printf("\t\tBinary initial squared gradients file to be loaded if -load-init-gradsq is 1; (default is to look for gradsq.000.bin)\n");
        printf("\t-seed <int>\n");
        printf("\t\tRandom seed to use.  If not set, will be randomized using current time.");
        printf("\nExample usage:\n");
        printf("./glove -input-file cooccurrence.shuf.bin -vocab-file vocab.txt -save-file vectors -gradsq-file gradsq -verbose 2 -vector-size 100 -threads 16 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2\n\n");
        result = 0;
    } else {
        if ((i = arg_pos((char *)"-write-header", argc, argv)) > 0) write_header = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-n", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-p", argc, argv))> 0) P = atoi(argv[i+1]);
        if ((i = arg_pos((char *)"-iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
        cost = (realglove *)malloc(sizeof(realglove) * num_threads);
        if ((i = arg_pos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
        if ((i = arg_pos((char *)"-x-max", argc, argv)) > 0) x_max = atof(argv[i + 1]);
        if ((i = arg_pos((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
        if ((i = arg_pos((char *)"-grad-clip", argc, argv)) > 0) grad_clip_value = atof(argv[i + 1]);
        if ((i = arg_pos((char *)"-binary", argc, argv)) > 0) use_binary = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-model", argc, argv)) > 0) model = atoi(argv[i + 1]);
        if (model != 0 && model != 1) model = 2;
        if ((i = arg_pos((char *)"-save-gradsq", argc, argv)) > 0) save_gradsq = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-vocab-file", argc, argv)) > 0) strcpy(vocab_file, argv[i + 1]);
        else strcpy(vocab_file, (char *)"vocab.txt");
        if ((i = arg_pos((char *)"-save-file", argc, argv)) > 0) strcpy(save_W_file, argv[i + 1]);
        else strcpy(save_W_file, (char *)"vectors");
        if ((i = arg_pos((char *)"-gradsq-file", argc, argv)) > 0) {
            strcpy(save_gradsq_file, argv[i + 1]);
            save_gradsq = 1;
        }
        else if (save_gradsq > 0) strcpy(save_gradsq_file, (char *)"gradsq");
        if ((i = arg_pos((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
        else strcpy(input_file, (char *)"cooccurrence.shuf.bin");
        if ((i = arg_pos((char *)"-checkpoint-every", argc, argv)) > 0) checkpoint_every = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-init-param-file", argc, argv)) > 0) strcpy(init_param_file, argv[i + 1]);
        else strcpy(init_param_file, (char *)"vectors.000.bin");
        if ((i = arg_pos((char *)"-load-init-param", argc, argv)) > 0) load_init_param = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-save-init-param", argc, argv)) > 0) save_init_param = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-init-gradsq-file", argc, argv)) > 0) strcpy(init_gradsq_file, argv[i + 1]);
        else strcpy(init_gradsq_file, (char *)"gradsq.000.bin");
        if ((i = arg_pos((char *)"-load-init-gradsq", argc, argv)) > 0) load_init_gradsq = atoi(argv[i + 1]);
        if ((i = arg_pos((char *)"-seed", argc, argv)) > 0) seed = atoi(argv[i + 1]);
        
        vocab_size = 0;
        fid = fopen(vocab_file, "r");
        if (fid == NULL) {log_file_loading_err("vocab file", vocab_file); free(cost); return 1;}
        while ((i = getc(fid)) != EOF) if (i == '\n') vocab_size++; // Count number of entries in vocab_file
        fclose(fid);
        if (vocab_size == 0) {fprintf(stderr, "Unable to find any vocab entries in vocab file %s.\n", vocab_file); free(cost); return 1;}
        result = train_glove();
        free(cost);
    }
    free(W);
    free(gradsq);

    return result;
}
