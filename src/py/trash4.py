# -*- coding: utf-8 -*-
"""
"""
from __future__ import division
from random import choice
import string
import math
import time
from itertools import product
from collections import defaultdict

import numpy as np
import pylab as pl

from common import art_plot, Line


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



def words_from_prefix(prefix):
    results = [{prefix}]
    for length in xrange(len(prefix), 3 * len(prefix)):
        result = set()
        for i, prefixes in enumerate(results[-len(prefix):]):
            for word in prefixes:
                result.add(word + prefix[:length + 1 - len(word)])
        results.append(result)
    return [len(i) for i in results]


def crazy_formula(prefix):
    border_array = border(prefix)
    border_array.reverse()
    temp = 1 - (np.array(border_array) > 0)
    result = [1]
    for _ in prefix * 2:
        result.append((temp[-min(len(result), len(prefix)):] * np.array(result[-len(prefix):])).sum())
    return result


def magic_pair(word):
    k = 1
    while k < len(word) and word[k] == word[0]:
        k += 1
    j = k
    while j < len(word) and word[j] != word[0]:
        j += 1
    return k, j - k


b_precise_table = defaultdict(lambda: defaultdict(int))


def b_precise(j, i, sigma):
    if j + 1 > i:
        return 0
    if b_precise_table[j][i] == 0:
        if i <= 2 * j + 2:
            result = (sigma - 1) ** (j - 1) * (sigma ** (i - j + 1) - 2 * sigma ** (i - j) + sigma)
        else:
            result = sigma * b_precise(j, i - 1, sigma)
            if i % 2 == 0:
                result -= b_precise(j, i / 2, sigma)

        b_precise_table[j][i] = result

    return b_precise_table[j][i]


b_0_precise_table = defaultdict(int)


def b_0_precise(i, sigma):
    if i < 1:
        return 0
    if i == 1:
        return sigma
    if b_0_precise_table[i] == 0:
        result = sigma * b_0_precise(i - 1, sigma)
        if i % 2 == 0:
            result -= b_0_precise(i / 2, sigma)
        b_0_precise_table[i] = result
    return b_0_precise_table[i]


delta_table = defaultdict(lambda: defaultdict(int))


def delta(i, j, sigma):
    if i == j:
        return b_0_precise(i, sigma)
    if j == 1:
        return b_0_precise(i, sigma)
    if j == 2:
        return b_precise(j, i, sigma)
    if delta_table[i][j] == 0:
        if 2 * j >= i:
            result = b_0_precise(j, sigma) * sigma ** (i - j) - (sigma ** i - sigma ** math.ceil(i / 2)) / (sigma - 1)
        else:
            result = delta(i - 1, j, sigma) * sigma
            if i % 2 == 0:
                result -= sigma ** (i / 2)
        delta_table[i][j] = result
    return delta_table[i][j]

c_table = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def c(j, k, i, sigma):
    if j + k > i:
        return 0

    if c_table[j][k][i] == 0:
        if j + k == i:
            result = sigma * (sigma - 1) ** j
        elif j + k + 1 == i:
            result = sigma * (sigma - 1) ** (j + 1)
        elif j + 2 * k + 1 > i:
            result = sigma ** (i - j - k) * (sigma - 1) ** (j + 1)
        elif (j + k) * 2 >= i:
            result = (sigma - 1) ** (j - 1) * (sigma ** (i - j - k) * (sigma - 1) ** 2 - sigma ** (i - j - 2 * k + 1) + sigma)
        else:
            result = sigma * c(j, k, i - 1, sigma)
            if i % 2 == 0:
                result -= c(j, k, i / 2, sigma)
        c_table[j][k][i] = result

    return c_table[j][k][i]


beta_table = defaultdict(lambda: defaultdict(int))


def beta(i, n, sigma):
    if i == n:
        return b_0_precise(i, sigma)
    if i > n:
        return 0
    if beta_table[i][n] == 0:
        result = b_0_precise(i, sigma) * beta(i, n - 1, sigma)
        if i != 1:
            result += b_0_precise(i, sigma) * beta(i, n - i, sigma)
        for j in xrange(2, i):
            result += delta(i, j, sigma) * beta(i, n - j, sigma)
        beta_table[i][n] = result
    return beta_table[i][n]

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
    # for word in all_words(AVAILABLE_LETTERS[:2], 10):
    #     assert crazy_formula(word) == words_from_prefix(word)
    #     if word[0] == 'B':
    #         break

    # max_val = 0
    # ALPHABET = 2
    # LENGTH = 30
    # alphabet = AVAILABLE_LETTERS[:ALPHABET]


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
    # max_lens = {
    #     2: 21,
    #     3: 15,
    #     4: 13,
    #     5: 12
    # }


    # for ALPHABET in xrange(2, 6):
    #     alphabet = AVAILABLE_LETTERS[:ALPHABET]
    #     result = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    #     prefixes = defaultdict(lambda: defaultdict(set))
    #     max_len = max_lens[ALPHABET]
    #     lengths = range(2, max_len)
    #     for n in lengths:
    #         print n
    #         for word in all_words(alphabet, n):
    #             i = max_borderless_prefix(word)
    #             if i < (n - 1) // 2 + 1:
    #                 continue
    #             prefix = word[:i]
    #             j = unequal_count_after_first_letter(prefix)
    #             for l in xrange(min(j, n - i - 1) + 1):
    #                 result[i][l][n] += 1
    #                 prefixes[i][l].add(prefix)
    #
    #     for i in result:
    #         for j in result[i]:
    #             lengths = sorted(result[i][j].iterkeys())
    #             bound = len(prefixes[i][j]) * (2 ** j)
    #             pl.plot(lengths, [result[i][j][n] for n in lengths], label='Words in form $SP_1...P_k$')
    #             pl.plot(lengths, [bound] * len(lengths), label='$b_j(i, \sigma) 2^j$', color='r', linewidth=2)
    #             pl.legend(loc=2)
    #             _, m = pl.axes().get_ylim()
    #             pl.axes().set_ylim((bound - 1, m))
    #             pl.savefig('lemma3/sigma{sigma}/i{i}_j{j}.png'.format(sigma=ALPHABET, i=i, j=j))
    #             pl.axes().clear()


            # pl.plot(lengths, [0] * j + [result[i][j] - ((ALPHABET - 1) ** (j + 1) * ALPHABET ** (i - j - 1) - ALPHABET ** (i - 2)) for i in lengths[j:]], label='difference')
            # pl.legend(loc=2)
            # pl.savefig('diff_sigma{}_j{}.png'.format(ALPHABET, j))
            # pl.axes().clear()

#     s = 3
#     for i in xrange(3, 13):
#         total = defaultdict(lambda: defaultdict(int))
#         available_letters = AVAILABLE_LETTERS[:s]
#         for word in all_words(available_letters, i):
#             if is_borderless(word):
#                 k, j = magic_pair(word)
#                 for l in xrange(1, j + 1):
#                     total[k][l] += 1
#         for k, values in total.iteritems():
#             for j, v in values.iteritems():
#                 if k > 1:
#                     assert c(j, k, i, s) == v
#
# else:
    max_lens = {
        # 2: 21,
        # 3: 21,
        # 4: 11,
        10: 5,
    }

    for alphabet, max_len in max_lens.iteritems():
        result = defaultdict(lambda: defaultdict(int))
        for length in xrange(1, max_len):
            start = time.time()
            print alphabet, length,
            available_letters = AVAILABLE_LETTERS[:alphabet]
            for word in all_words(available_letters, length):
                if word[0] != 'A':
                    break
                if is_borderless(word):
                    j = unequal_count_after_first_letter(word)
                    for l in xrange(j + 1):
                        result[l][length] += 1
            print int(time.time() - start), 'sec'

        for j, values_per_length in result.iteritems():
            q = alphabet
            for length in sorted(values_per_length):
                if j == 0:
                    continue
                my_value = q ** (length - j) * (q - 1) ** (j - 1) * (q - 2)
                assert my_value < values_per_length[length] * q

            lengths = sorted(values_per_length)
            plot_error = False
            if plot_error:
                if j == 1 or j == 5:
                    print '=' * 40
                    print 'sigma =', q, 'j =', j
                    for l in lengths:
                        print l, '&', '{:.3f}'.format((values_per_length[l] * q - q ** (l - j) * (q - 1) ** (j - 1) * (q - 2)) / (values_per_length[l] * q)), '\\\\ \\hline'
                    print '=' * 40
                lines = [
                    Line(
                        lengths,
                        [(values_per_length[l] * q - q ** (l - j) * (q - 1) ** (j - 1) * (q - 2)) / (values_per_length[l] * q) for l in lengths],
                        'Relative error'
                    ),
                ]
                art_plot(
                    lines=lines,
                    filename='b_j_results/{}/{}err{}.pdf'.format(q, q, j),
                    font_size=19,
                    xlabel=u'Длина слова $i$',
                    xlim=(min(lengths), max(lengths)),
                )
            else:
                lines = [
                    Line(
                        xrange(min(lengths), 21),
                        [q ** (l - j) * (q - 1) ** (j - 1) * (q - 2) / q ** l for l in xrange(min(lengths), 21)],
                        '$\sigma^{-j}(\sigma - 1)^{j - 1}(\sigma - 2)$',
                    ),
                    Line(
                        xrange(min(lengths), 21),
                        [values_per_length.get(l) * q / q ** l for l in lengths],
                        '$v_j(i, \sigma)$',
                    )
                ]

                line = lines[1]
                lines[1] = Line(
                    line[0],
                    line[1] + [line[1][-1]] * (21 - min(line[0]) - len(line[1])),
                    line[2],
                )
                print 'j = ', j
                print '\n'.join(map(str, [(l, values_per_length.get(l) * q / q ** l) for l in lengths]))
                art_plot(
                    lines=lines,
                    filename='b_j_results/{}/{}j{}.pdf'.format(q, q, j),
                    font_size=19,
                    xlabel=u'Длина слова $i$',
                    legend_location=0,
                    xlim=(min(lengths), 20),
                    mirror_ymin=q == 2,
                )
