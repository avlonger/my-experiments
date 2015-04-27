#include <stdlib.h>
#include <string.h>
#include <stdbool.h>


int duval(const char * text, int * output){
    size_t n = strlen(text);
    int h = 0;
    int words_count = 0;
    while (h < n) {
        int i = h;
        int j = h + 1;
        while (text[j] >= text[i]) {
            if (text[j] > text[i])
                i = h;
            else
                i++;
            j++;
        }
        while (h <= i) {
            output[words_count++] = h;
            h += j - i;
        }
    }
    return words_count;
}


int naive_lyndon_decomposition(const char * text, int * output) {
    size_t n = strlen(text);

    if (n == 1) {
        output[0] = 0;
        return 1;
    }

    int * next_word = calloc(n, sizeof(int));

    // build first (not Lyndon) decomposition of size n
    for (int i = 0; i < n; ++i) {
        next_word[i] = i + 1;
    }

    // concatenate consequent factors if first is less than the second
    bool progress = true;
    while (progress) {
        progress = false;
        int first_word_start = 0;
        int second_word_start = next_word[first_word_start];
        while (second_word_start < n) {
            int third_word_start = next_word[second_word_start];

            int first_word_index = first_word_start;
            int second_word_index = second_word_start;
            while (first_word_index < second_word_start && second_word_index < third_word_start && text[first_word_index] == text[second_word_index]) {
                first_word_index++;
                second_word_index++;
            }

            if (second_word_index < third_word_start && (first_word_index == second_word_start || text[first_word_index] < text[second_word_index])) {
                progress = true;
                next_word[first_word_start] = third_word_start;
            }

            first_word_start = next_word[first_word_start];
            second_word_start = (first_word_start < n) ? next_word[first_word_start] : (int)n;
        }
    }
    int word_count = 0;
    int word_start = 0;
    while (word_start < n) {
        output[word_count++] = word_start;
        word_start = next_word[word_start];
    }

    free(next_word);
    return word_count;
}
