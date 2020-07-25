// discretize a given word vector
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define max_size 2000
#define N 40
#define max_w 50

typedef float real;

FILE *f;
char st1[max_size];
char *bestw[N];
char file_name[max_size], out_file_name[max_size], st[100][max_size];
float dist, len, bestd[N], vec[max_size];
long long words, size, a, b, c, d, cn, bi[100];
float *M;
char *vocab;

#define NUM_SPARKS 100

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("usage: %s <input-bin-file> <output-text-file-for-snake>\n", argv[0]);
    return 1;
  }
  FILE *f;

  strcpy(file_name, argv[1]);
  strcpy(out_file_name, argv[2]);

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
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
        (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {

    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;
    for (a = 0; a < size; a++) {
        fread(&M[a + b * size], sizeof(float), 1, f);
        // need to discretize
        M[a + b * size];
    }
  }


  fclose(f);
  f = fopen(out_file_name, "wb");
  fprintf(f, "%lld %lld\n", words, size);
  for (b = 0; b < words; b++) {
      fprintf(fo, "%s ", vocab[a].word);
      for (int i = 0; i < layer1_size; i++) {
          fwrite(&M[a * size + b], sizeof(real), 1, fo);
      }
  }
  fclose(f);
  return 0;
}
