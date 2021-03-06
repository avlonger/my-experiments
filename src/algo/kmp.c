#include <stdlib.h>
#include <string.h>

void border(const char * text, int * border) {
    size_t n = strlen(text);
    border[0] = 0;
    int j = 0;
    for (int i = 1; i < n; ++i) {
        j = border[i - 1];
        while (j > 0 && text[i] != text[j]) {
            j = border[j - 1];
        }
        if (text[i] == text[j])
            j++;
        border[i] = j;
    }
}

int kmp(const char * pattern, const char * text, int * output){
    size_t m = strlen(pattern), n = strlen(text);
    int * pattern_border = calloc(m, sizeof(int));
    border(pattern, pattern_border);
    int i = 0, k = 0;
    for (int j = 0; j < n; ++j) {
        while (i > 0 && pattern[i] != text[j])
            i = pattern_border[i - 1];
        if (pattern[i] == text[j])
            i++;
        if (i >= m) {
            output[k++] = j - i + 1;
            i = pattern_border[i - 1];
        }
    }
    free(pattern_border);
    return k;
}
