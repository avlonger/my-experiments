#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "algo/kmp.h"


int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s TEXT\n", argv[0]);
        return 1;
    }
    char * text = argv[1];

    int * output = calloc(strlen(text), sizeof(int));

    border(text, output);

    printf("%d\n", output[strlen(text) - 1]);

    return 0;
}
