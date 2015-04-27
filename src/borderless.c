#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "algo/borderless.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s TEXT\n", argv[0]);
        return 1;
    }
    char * text = argv[1];

    int start;

    int size = longest_borderless_subword(text, &start);

    char result[size + 1];

    printf("LENGTH: %d\n", size);

    memcpy(result, text + start, size * sizeof(char));
    result[size] = 0;

    printf("%s\n", result);

    return 0;
}
