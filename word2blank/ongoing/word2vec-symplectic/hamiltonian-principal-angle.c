// Attempt to recreate 

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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

float func(float a, float b);           // f(distance, momentum)

static const long long max_size = 2000;         // max length of strings
const long long N = 10;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

// orthonormalize a2 wrt a1
void orthonorm(const int size, const float *a1, const float *a2, float *a2_perp) {
    // we assume a1 is orthonormal.
    float dot = 0;
    for(int i = 0; i < size; ++i) {
        dot += a1[i] * a2[i];
    }

    // compute a2' = a2 - (a2.a1) a1_hat
    // we assume that a1 is already normalized.
    float l = 0;
    for(int i = 0; i < size; ++i) {
        a2_perp[i] = a2[i] -  dot * a1[i];
        l += a2_perp[i] * a2_perp[i];
    }

    l = sqrt(l);
    for(int i = 0; i < size; ++i) { a2_perp[i] /= l; }


}

void principal_angles(const int size,
        const float *a1, const float *a2,
        const float *b1, const float *b2,
        float *sigma_1, float *sigma_2) {
    float a2_perp[max_size];
    float b2_perp[max_size];

    orthonorm(size, a1, a2, a2_perp);
    orthonorm(size, b1, b2, b2_perp);
    float mat[2][2];
    for(int i = 0; i < 2; ++i) {
        const float *a = i == 0 ? a1 : a2_perp;
        for(int j = 0; j < 2; ++j) {
            const float *b = j == 0 ? b1 : b2_perp;
            mat[i][j] = 0;
            for(int k = 0;  k < size; ++k) { mat[i][j] += a[k] * b[k]; }
        } // end j
    } // end i

    // SVD of a 2x2 matrix
    const float a = mat[0][0], b = mat[0][1], c = mat[1][0], d = mat[1][1];
    const float s1 = a*a + b*b + c*c + d*d;
    const float s2 = 
        sqrt((a*a + b*b - c*c - d*d) * (a*a + b*b - c*c - d*d) + 
                (4 * (a*c + b*d) * (a*c + b*d))); 
    *sigma_1 = sqrt(0.5*(s1 + s2)); 
    *sigma_2 = (s1 <= s2 ? 0 : sqrt(0.5*(s1 - s2))); 
    //*sigma_1_sq = 0.5*(s1 + s2); 
    //*sigma_2_sq = 0.5*(s1 - s2);
}


int main(int argc, char **argv) {
    srand(0);
    FILE *f;
    char st1[max_size];
    char *bestw[N];
    char file_name[max_size], st[100][max_size];
    float dist, mom, len, disp, delmom, deldist, bestd[N], vec[max_size];
    long long words, size, a, b, c, d, cn, bi[100];
    float *M;
    float *lens;
    char *vocab;
    if (argc < 2) {
        printf("Usage: ./hamiltonian <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
        return 0;
    }
    strcpy(file_name, argv[1]);
    f = fopen(file_name, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (float *)malloc((long long)words * (long long)size * sizeof(float));
    lens = (float *)malloc((long long)words * (long long)2 * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }
    // Read Vectors 
    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        vocab[b * max_w + a] = 0;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
      
        // Normalize `p`
        len = 0;
        for (a = 0; a < size/2; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size/2; a++) M[a + b * size] /= len;
        
        // Normalize `q`
        len = 0;
        for (a = size/2; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = size/2; a < size; a++) M[a + b * size] /= len;
        
        for(a = 0; a < size; a++) {
            if (M[a + b * size] != M[a + b * size]) { 
                printf("nan |%20s|[%4lld]\n", vocab + b * max_w, a);
            }
        }
    }
    fclose(f);

    long long NWORDS = 5; // number of words being studied for this purpose
    double dt = 1e-3; // value of dt
    long long NITERS = 10; // number of iterations
    long long posns[NWORDS];
    for (long long j = 0; j < NWORDS; j++)  posns[j] = 1 + rand() % (words-1);    // random position for a given word

    for (long long i = 0; i < NITERS; i++)  {
      for (b = 0; b < words; b++) {
        for (a = 0; a < size/2; a++) M[(a + size/2) + b * size] += M[a + b * size] * dt;  // dq += p * dt
      }
      // finding similar words
      for (long long j = 0; j < NWORDS; j++)  { 
        printf("\nWord: %s\n", vocab + posns[j] * max_w);

        float *vec = M + posns[j] * size;
        
        // dist = 0, mom = 0;
        // for (a = 0; a < size/2; a++) dist += vec[a] * vec[a];
        // for (a = size/2 ; a < size; a++) mom += vec[a] * vec[a];
        // dist = sqrt(dist);
        // mom = sqrt(mom);
        // for (a = 0; a < size/2; a++) vec[a] /= dist;
        // for (a = size/2; a < size; a++) vec[a] /= mom;
        
        for (a = 0; a < N; a++) bestd[a] = -1;
        for (a = 0; a < N; a++) bestw[a][0] = 0;
        
        for (c = 0; c < words; c++) {
          a = 0;
          if (posns[j] == c) a = 1;
          if (a == 1) continue;
  
          
          float ct_1 = 0, ct_2 = 0;
          principal_angles(size/2, vec, vec+size/2,
                  M + c*size, M + c*size + size/2,
                  &ct_1, &ct_2);
          const float gain = ct_1*ct_1 + ct_2*ct_2;
          
          
          for (a = 0; a < N; a++) {
            if (gain > bestd[a]) {
              for (d = N - 1; d > a; d--) {
                bestd[d] = bestd[d - 1];
                strcpy(bestw[d], bestw[d - 1]);
              }
              bestd[a] = gain;
              strcpy(bestw[a], &vocab[c * max_w]);
              break;
            }
          }
        }
        for (a = 0; a < N; a++) printf("%30s\n", bestw[a]);
      }
      printf("\n\n------------------------------------\n\n");
    }
    return 0;
}
