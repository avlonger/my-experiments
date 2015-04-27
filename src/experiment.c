#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include "algo/borderless.h"
#include "algo/kmp.h"


// some buffers
char * CHAR_BUFFER;
int * INT_BUFFER;
int (* FUNCTION)();
char ALPHABET = 2;
char MINIMAL_CHAR = 1;
int LENGTH = 2;
bool TRACE = false;
bool NORMALIZE = false;
bool PRINT_TOTAL = false;
char PREFIX = 0;
char PREFIX_LENGTH = 0;


void usage(const char * program_name) {
    printf("Usage: %s [options] ALGORITHM\n", program_name);
    printf("Calculate some value for words range\n");
    printf("Algorithms:\n");
    printf("MINPERIOD               find minimal period of a word\n");
    printf("MAXBORDERLESS           find longest borderless subword\n");
    printf("PERIOD_BORDERLESS_DIFF  find difference between min period\n");
    printf("                        and longest borderless subword\n\n");
    printf("Options:\n");
    printf(" -l  Find average value for all words of given length\n");
    printf(" -a  Alphabet size (default: 2)\n");
    printf(" -t  Trace: print results for all generated words\n");
    printf(" -n  Normalize (divide result by total words count)\n");
    printf(" -c  Print count of processed words\n");
    printf(" -f  Fixed prefix length\n");
    printf(" -p  Prefix decimal representation\n");
}


uint64_t do_for_all_words(int position) {
    uint64_t total = 0;
    int value = 0;
    for (char i = MINIMAL_CHAR; i < ALPHABET; ++i) {
        CHAR_BUFFER[position] = i;

        if (position < LENGTH - 1) {
            total += do_for_all_words(position + 1);
        } else {
            value = FUNCTION();

            if (TRACE) {
                printf("%s\t%d\n", CHAR_BUFFER, value);
            }

            total += value;
        }
    }

    return total;
}


int minimal_period() {
    border(CHAR_BUFFER, INT_BUFFER);
    return LENGTH - INT_BUFFER[LENGTH - 1];
}


int longest_borderless() {
    return longest_borderless_subword(CHAR_BUFFER, INT_BUFFER);
}


int longest_borderless_minimal_period_diff() {
    int period = minimal_period();
    if (period == LENGTH)
        return 0;
    return period - longest_borderless();
}


int main(int argc, char** argv) {
    int c = 0;
    opterr = 1;
    while ((c = getopt(argc, argv, "tncl:a:p:f:")) != -1)
        switch (c)
        {
            case 'a':
                ALPHABET = (char) atoi(optarg);
                break;
            case 'l':
                LENGTH = atoi(optarg);
                break;
            case 't':
                TRACE = true;
                break;
            case 'n':
                NORMALIZE = true;
                break;
            case 'c':
                PRINT_TOTAL = true;
                break;
            case 'p':
                PREFIX = (char) atoi(optarg);
                break;
            case 'f':
                PREFIX_LENGTH = (char) atoi(optarg);
                break;
            case '?':
            case ':':
                return -1;
            default:
                usage(argv[0]);
                return -1;
        }

    if (TRACE) {
        MINIMAL_CHAR = 'A';
    }

    if (optind >= argc) {
        usage(argv[0]);
        return -1;
    }

    if (strcmp(argv[optind], "MINPERIOD") == 0) {
        FUNCTION = minimal_period;
    } else if (strcmp(argv[optind], "MAXBORDERLESS") == 0) {
        FUNCTION = longest_borderless;
    } else if (strcmp(argv[optind], "PERIOD_BORDERLESS_DIFF") == 0) {
        FUNCTION = longest_borderless_minimal_period_diff;
    } else {
        usage(argv[0]);
        return -1;
    }

    CHAR_BUFFER = calloc((size_t) LENGTH + 1, sizeof(char));
    INT_BUFFER = calloc((size_t) LENGTH + 1, sizeof(int));
    char * temp_buffer = calloc((size_t) PREFIX_LENGTH + 1, sizeof(char));

    int filled_chars = 0;
    while (PREFIX > 0) {
        temp_buffer[filled_chars++] = PREFIX % ALPHABET;
        PREFIX /= ALPHABET;
    }

    for (int i = 0; i < PREFIX_LENGTH; ++i) {
        CHAR_BUFFER[i] = temp_buffer[PREFIX_LENGTH - i - 1] + MINIMAL_CHAR;
    }

    ALPHABET += MINIMAL_CHAR;

    uint64_t answer = do_for_all_words(PREFIX_LENGTH);

    double count = pow((double) (ALPHABET - MINIMAL_CHAR), (double) LENGTH);
    if (NORMALIZE) {
        printf("%.10f\n", (double) answer / count);
    } else {
        printf("%llu\n", answer);
    }

    if (PRINT_TOTAL) {
        printf("%d\n", (int)count);
    }

    free(CHAR_BUFFER);
    free(INT_BUFFER);
    return 0;
}
