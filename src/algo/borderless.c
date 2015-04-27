#include <string.h>
#include <stdlib.h>
#include "kmp.h"

int longest_borderless_subword(const char * text, int * output) {
    // TODO: print all longest borderless subwords??
    int n = strlen(text);
    int * border_buffer = calloc(n, sizeof(int));
    int max_len = 1, maxi = 0;
    for (int i = 0; i < n; ++i) {
        border(text + i, border_buffer + i);
        for (int j = n - 1; j > i && j - i + 1 > max_len; --j) {
            if (border_buffer[j] == 0) {
                max_len = j - i + 1;
                maxi = i;
            }
        }
    }
    output[0] = maxi;
    free(border_buffer);
    return max_len;
}
