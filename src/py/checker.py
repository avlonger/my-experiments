# -*- coding: utf-8 -*-
"""
"""
import string
import subprocess
from random import choice
from itertools import product


import pylab as pl
import numpy as np

EXPERIMENTS = 1000
LENGTH = 100
AVAILABLE_LETTERS = string.lowercase + string.uppercase + string.digits


def random_string(length, alphabet_size):
    return ''.join(choice(AVAILABLE_LETTERS[:alphabet_size]) for _ in xrange(length))


def is_lyndon_word(word):
    for i in xrange(1, len(word)):
        if word >= word[i:]:
            return False
    return True


def is_lyndon_decomposition(word, factor_positions):
    slice_positions = list(factor_positions)
    slice_positions.append(len(slice_positions))
    factors = []

    # testing if each factor is Lyndon-word
    for i, position in enumerate(slice_positions[:-1]):
        factor = word[position:slice_positions[i + 1]]
        if not is_lyndon_word(factor):
            return False
        factors.append(factor)

    # testing if each factor is not less than the next factor
    for i, factor in enumerate(factors[:-1]):
        if not factor >= factors[i + 1]:
            return False

    return True


def is_longest_borderless(text, length, answer):
    n = len(text)
    max_len = 1
    for i in xrange(n):
        for j in xrange(i + 1, n + 1):
            subword = text[i:j]
            for k in xrange(1, len(subword) - 1):
                subsub = subword[:k]
                if subword.endswith(subsub):
                    break
            else:
                max_len = max(len(subword), max_len)
    return max_len == length


def all_words(alphabet, length):
    for result in product(alphabet, repeat=length):
        yield ''.join(result)

if __name__ == '__main__':

    # print 'Testing borderless algorithm in series of {} experiments...'.format(EXPERIMENTS)
    # for _ in xrange(EXPERIMENTS):
    #     print _
    #     text = random_string(LENGTH, len(AVAILABLE_LETTERS))
    #     process = subprocess.Popen(['../../bin/borderless', text], stdout=subprocess.PIPE)
    #     output = process.stdout.read().strip()
    #     length = int(output.splitlines()[0].split()[-1].strip())
    #     subword = output.splitlines()[1].strip()
    #     assert is_longest_borderless(text, length, subword)
    #     print length
    # print 'OK'

    for alphabet_size in xrange(2, 5):
        average_values = []
        lengths = range(2, 9)
        for length in lengths:
            print 'Alphabet:', alphabet_size, 'Length:', length
            values = []
            for i, text in enumerate(all_words(AVAILABLE_LETTERS[:alphabet_size], length)):
                if i % 1000 == 0:
                    print i
                process = subprocess.Popen(['../../bin/borderless', text], stdout=subprocess.PIPE)
                output = process.stdout.read().strip()
                words_count = int(output.splitlines()[0].split()[-1].strip())
                values.append(words_count)
            average_values.append(np.average(values))
        print 'OK'
        pl.plot(lengths, average_values, marker='.', label='$\sigma = {}$'.format(alphabet_size))
    pl.xlabel('Text length')
    pl.ylabel('Longest borderless subword')
    pl.grid()
    pl.legend(loc=4)
    pl.savefig('../../results/borderless-alphabets.png')
    pl.clf()

    # print 'Testing duval algorithm in series of {} experiments...'.format(EXPERIMENTS)
    # for _ in xrange(EXPERIMENTS):
    #     text = random_string(LENGTH, len(AVAILABLE_LETTERS))
    #     process = subprocess.Popen(['../../bin/duval', text], stdout=subprocess.PIPE)
    #     output = process.stdout.read().strip()
    #     factor_positions = map(int, output.splitlines()[1].split())
    #     assert is_lyndon_decomposition(text, factor_positions)
    # print 'OK'

    # alphabet_sizes = range(2, len(AVAILABLE_LETTERS) + 1)
    # for length in [10, 100, 500, 999]:
    #     average_values = []
    #     print 'Plotting for length: {}'.format(length)
    #     for alphabet_size in alphabet_sizes:
    #         values = []
    #         for _ in xrange(EXPERIMENTS / 10):
    #             text = random_string(length, alphabet_size)
    #             process = subprocess.Popen(['../../bin/duval', text], stdout=subprocess.PIPE)
    #             output = process.stdout.read().strip()
    #             words_count = int(output.splitlines()[0].split()[-1].strip())
    #             factor_positions = map(int, output.splitlines()[2].split())
    #             assert is_lyndon_decomposition(text, factor_positions)
    #             values.append(words_count)
    #         average_values.append(sum(values) * 1.0 / len(values))
    #     pl.plot(alphabet_sizes, average_values)
    #     pl.title('Text length = {}'.format(length))
    #     pl.xlabel('Alphabet size')
    #     pl.ylabel('Factors count')
    #     pl.savefig('../../results/py-length-{}.png'.format(length))
    #     pl.clf()

        # alphabet_sizes = range(2, len(AVAILABLE_LETTERS) + 1)

