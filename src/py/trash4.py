# -*- coding: utf-8 -*-
"""
"""
from __future__ import division
from random import choice
import string
import math
from itertools import product
from collections import defaultdict

import numpy as np
import pylab as pl


AVAILABLE_LETTERS = string.uppercase


def all_words(alphabet, length):
    for result in product(alphabet, repeat=length):
        yield ''.join(result)


def border(text):
    n = len(text)
    array = [0]
    for i in xrange(1, n):
        j = array[i - 1]
        while j > 0 and text[i] != text[j]:
            j = array[j - 1]
        if text[i] == text[j]:
            j += 1
        array.append(j)
    return array


def max_borderless_prefix(text):
    border_buffer = border(text)
    j = len(text)
    while border_buffer[j - 1] != 0:
        j -= border_buffer[j - 1]
    return j


def max_borderless_suffix(text):
    return max_borderless_prefix(text[::-1])


def is_borderless(text):
    return border(text)[-1] == 0


def line_cut_count(text):
    border_buffer = border(text)
    j = len(text)
    step = 0
    while border_buffer[j - 1] != 0:
        j -= border_buffer[j - 1]
        step += 1
    return step


def max_borderless(text):
    a, b = max_borderless_positions(text)
    return b - a


def max_borderless_prefix(text):
    border_buffer = border(text)
    j = len(text)
    while border_buffer[j - 1] != 0:
        j -= border_buffer[j - 1]
    return j


def query_count_except_first(text):
    n = len(text)

    max_len = 1

    i = 0

    cnt = 0

    while i < n and n - i > max_len:

        border_buffer = border(text[i:])

        j = n - i

        while j > max_len:

            if border_buffer[j - 1] == 0:
                max_len = j

            if j != n - i:
                cnt += 1

            j -= border_buffer[j - 1]

        i += 1

    return cnt


def jumps_per_suffix(text):
    n = len(text)

    max_len = 1

    i = 0

    jumps = []

    while i < n and n - i > max_len:

        jumps.append(0)

        border_buffer = border(text[i:])

        j = n - i

        while j > max_len:

            if border_buffer[j - 1] == 0:
                max_len = j

            jumps[-1] += 1

            j -= border_buffer[j - 1]

        jumps[-1] -= 1

        i += 1

    return jumps


def query_count(text):
    n = len(text)

    max_len = 1

    i = 0

    cnt = 0

    while i < n and n - i > max_len:

        border_buffer = border(text[i:])

        j = n - i

        while j > max_len:

            if border_buffer[j - 1] == 0:
                max_len = j

            cnt += 1

            j -= border_buffer[j - 1]

        i += 1

    return cnt


def max_borderless_positions(text):
    n = len(text)

    max_len = 1

    positions = (0, 1)

    i = 0

    while i < n and n - i > max_len:

        border_buffer = border(text[i:])

        j = n - i - border_buffer[-1]

        while j > max_len:

            if border_buffer[j - 1] == 0:
                max_len = j
                positions = (i, j + i)

            j -= border_buffer[j - 1]

        i += 1

    return positions


def max_borderless_with_debug(text):
    n = len(text)
    max_len = 1

    i = 0

    # q = 0

    # results = []

    # to_print = False

    while i < n and n - i > max_len:

        border_buffer = border(text[i:])

        # results.append([0] * i + border_buffer)

        j = n - i - border_buffer[-1]

        # cuts = []

        while j > max_len:

            # q += 1

            # cuts.append(j + i)

            if border_buffer[j - 1] == 0:
                max_len = j
            # else:
                # second_cut = border_buffer[j - 1]
                # if second_cut > first_cut:
                #     to_print = True

            j -= border_buffer[j - 1]

        return j
        # cuts.append(j + i)


        # if i > 0:
        #     result = [' ' for _ in xrange(i)]
        # else:
        #     result = []
        #
        # for cnt, c in enumerate(text[i:]):
        #     cnt += i
        #
        #     if cnt in cuts:
        #         result.append('|')
        #     result.append(c)
        #
        # results.append(result)

        i += 1
    # if to_print:

    #     for result in results:
    #         print ''.join(result)

    # for row, result in enumerate(results):
    #     for col, val in enumerate(result):
    #         if row > 0 and results[row - 1][col] - 1 > val:
    #             to_print = True
    # if to_print:
    #     print '-' * 10
    #     for result in results:
    #         print result
    return max_len


def random_string(length, alphabet):
    return ''.join(choice(alphabet) for _ in xrange(length))


def magic_word(k, m):
    return 'A' * k + 'BB' + ('A' * (k + 1) + 'B') * m


def unequal_count_after_first_letter(word):
    for i in xrange(1, len(word)):
        if word[i] == word[0]:
            return i - 1
    return len(word) - 1


if __name__ == '__main__':
    # magic = magic_word(20, 10)
    # q_magic = query_count(magic)
    # print q_magic, magic
    # print
    # l = len(magic)
    # q_max = 0
    # for b_pos in xrange(l):
    #     word = 'A' * b_pos + 'B' + 'A' * (l - b_pos - 1)
    #     q = query_count(word)
    #     if q > q_max:
    #         q_max = q
    #     print q, word


    max_val = 0
    ALPHABET = 2
    LENGTH = 30
    alphabet = AVAILABLE_LETTERS[:ALPHABET]
    # for length in xrange(6, LENGTH):
    #     print '\nLength', length
    #     for b_pos in xrange(1, length / 2 - 1):
    #         prefix = 'A' * b_pos + 'B'
    #         max_val = 0
    #         maxi = []
    #         for suffix in all_words(alphabet, length - len(prefix)):
    #             word = prefix + suffix
    #             val = query_count(word)
    #             if val == max_val:
    #                 maxi.append(word)
    #             elif val > max_val:
    #                 max_val = val
    #                 maxi = [word]
    #         if prefix + 'A' * (length - len(prefix)) not in maxi:
    #             print '=' * 10
    #             print prefix, ':'
    #             for w in maxi:
    #                 print w, 'period:', length - border(w)[-1]
    #             print 'Words above require queries:', max_val
    #             our_word = prefix + 'A' * (length - len(prefix))
    #             print our_word, 'period:', length - border(our_word)[-1]
    #             print 'Word above requires queries:', query_count(our_word)
    # k = 5
    # w = 'A' * k + 'B' * (k * 2) + 'A' * (k + 1) + 'B' * (k * 2) + 'A' * k + 'B' * k
    # print jumps_per_suffix(w)
#     k = 10
#     # w = 'AB' + 'A' * k + 'BAB' + 'A' * k + 'B' + 'A' * k + 'B'
#     m = 101
#     w = 'A' * (k + 1) + 'B' + 'A' * ((k + 1) * m)
#     jumps = jumps_per_suffix(w)
#     print jumps[0], jumps[k]
#     # jumps[1] = m
#     # jumps[k + 1] = (k + 1) * jumps[1] - k - 1
#     (k + 1) * jumps[0] - k - 1
# else:
#     ALPHABET = 2
#     LENGTH = 30
#     alphabet = AVAILABLE_LETTERS[:ALPHABET]
#     for length in xrange(2, LENGTH):
#         print '\nLength', length
#         max_val = 0
#         maxi = []
#         for word in all_words(alphabet, length):
#             if word[0] == 'B':
#                 break
#             jumps = jumps_per_suffix(word)
#             if len(jumps) > 1:
#                     for i, j in enumerate(jumps):
#                         if i > 0 and j < jumps[i - 1] * 0.5:
#                             break
#                     else:
#                         print '=' * 40
#                         print word
#                         print '-'.join(map(str, jumps))

    # ALPHABET = 2
    # LENGTH = 30
    # alphabet = AVAILABLE_LETTERS[:ALPHABET]
    # for length in xrange(LENGTH, LENGTH + 1):
    #     total = 0
    #     for word in all_words(alphabet, length):
    #         if word[0] == 'B':
    #             print ALPHABET, length, total * ALPHABET
    #             break
    #         # val = max_borderless_prefix(word)
    #         # print word, val
    #         total += max_borderless_prefix(word)
    # ALPHABET = 2
    # LENGTH = 30
    # alphabet = AVAILABLE_LETTERS[:ALPHABET]
    # for length in xrange(LENGTH, LENGTH + 1):
    #     total = 0
    #     i = 0
    #     while True:
    #         word = random_string(length, alphabet)
    #         total += max_borderless(word)
    #         i += 1
    #         if i % 1000 == 0:
    #             print total / i
    max_lens = {
        2: 21,
        3: 15,
        4: 13,
        5: 12
    }
    for ALPHABET in xrange(2, 6):
        alphabet = AVAILABLE_LETTERS[:ALPHABET]
        result = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        prefixes = defaultdict(lambda: defaultdict(set))
        max_len = max_lens[ALPHABET]
        lengths = range(2, max_len)
        for n in lengths:
            print n
            for word in all_words(alphabet, n):
                i = max_borderless_prefix(word)
                if i < (n - 1) // 2 + 1:
                    continue
                prefix = word[:i]
                j = unequal_count_after_first_letter(prefix)
                for l in xrange(min(j, n - i - 1) + 1):
                    result[i][l][n] += 1
                    prefixes[i][l].add(prefix)

        for i in result:
            for j in result[i]:
                lengths = sorted(result[i][j].iterkeys())
                bound = len(prefixes[i][j]) * (2 ** j)
                pl.plot(lengths, [result[i][j][n] for n in lengths], label='Words in form $SP_1...P_k$')
                pl.plot(lengths, [bound] * len(lengths), label='$b_j(i, \sigma) 2^j$', color='r', linewidth=2)
                pl.legend(loc=2)
                _, m = pl.axes().get_ylim()
                pl.axes().set_ylim((bound - 1, m))
                pl.savefig('lemma3/sigma{sigma}/i{i}_j{j}.png'.format(sigma=ALPHABET, i=i, j=j))
                pl.axes().clear()


            # pl.plot(lengths, [0] * j + [result[i][j] - ((ALPHABET - 1) ** (j + 1) * ALPHABET ** (i - j - 1) - ALPHABET ** (i - 2)) for i in lengths[j:]], label='difference')
            # pl.legend(loc=2)
            # pl.savefig('diff_sigma{}_j{}.png'.format(ALPHABET, j))
            # pl.axes().clear()
