#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#define DEBUG(x) if(0){ x; };

const long long max_size = 2000;         // max length of strings
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char *argv[]) {
    if (argc != 3) { 
        fprintf(stderr, "usage: %s <IN:TEXTPATH> <OUT:BINPATH>\n", argv[0]);
    }

    FILE *f;
    f = fopen(argv[1], "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }

    FILE *out;
    out = fopen(argv[2], "wb");
    if (f == NULL) {
        printf("Unable to open output file: |%s|\n", argv[2]);
        return -1;
    }

    long long words; 
    fscanf(f, "%lld", &words);
    long long size;
    fscanf(f, "%lld", &size);

    fprintf(out, "%lld %lld\n", words, size);

    char vocab[max_w];
    float vec[max_size];

  for (int w = 0; w < words; ++w) {
      // read word
      int a = 0;
      while(1) {
          vocab[a] = fgetc(f);
          if (feof(f) || (vocab[a] == ' ')) break;
          a++;
      }
      vocab[a] = '\0';

    fprintf(out, "%s ", vocab);
    DEBUG(printf("%s ", vocab))

      for(int i = 0; i < size; ++i) {
          char valstr[1024];
          a = 0;
          while(1) {
              valstr[a] = fgetc(f);
              if (feof(f) || valstr[a] == ' ' || valstr[a] == '\n') break;
              a++;
          }
          valstr[a] = '\0';
          vec[i] = atof(valstr);
          DEBUG(printf("%f ", vec[i]));
          fwrite(vec + i, 1, sizeof(float), out);
      }
      DEBUG(printf("\n"));
      fprintf(out, "\n");


      // read till a newline or EOF
      while(fgetc(f) != '\n' && !feof(f)) {};
  }

  fclose(f);
  fclose(out);

}
