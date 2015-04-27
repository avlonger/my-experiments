#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "algo/kmp.h"


int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s PATTERN TEXT\n", argv[0]);
        return 1;
    }
    char * pattern = argv[1];
    char * text = argv[2];

    int * output = calloc(strlen(text), sizeof(int));

    int occ = kmp(pattern, text, output);

    printf("OCCURENCES: %d\n", occ);

    for (int i = 0; i < occ; ++i) {
        printf("%d ", output[i]);
    }

    printf("\n");

    return 0;
}
