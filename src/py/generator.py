# -*- coding: utf-8 -*-
"""
"""
import string
import subprocess
from random import choice
from itertools import product

MIN_LEN = 14
MAX_LEN = 28
EXPERIMENTS = 50
LENGTH = 100
AVAILABLE_LETTERS = string.uppercase


def all_words(alphabet, length):
    for result in product(alphabet, repeat=length):
        yield ''.join(result)

def random_string(length, alphabet_size):
    return ''.join(choice(AVAILABLE_LETTERS[:alphabet_size]) for _ in xrange(length))

if __name__ == '__main__':
    # printed = 0
    # print 'Generating words and printing non-borderless words only...'
    # print 'WORD'.ljust(MAX_LEN + 3), 'BORDERLESS SUBWORD'.ljust(MAX_LEN + 3), 'PERIOD'
    # generated = 0
    # while printed < EXPERIMENTS:
    #     for length in xrange(MIN_LEN, MAX_LEN + 1):
    #         generated += 1
    #         s = random_string(length, len(AVAILABLE_LETTERS))
    #         process = subprocess.Popen(['../../bin/border', s], stdout=subprocess.PIPE)
    #         border_length = int(process.stdout.read().strip())
    #         period = length - border_length
    #         process = subprocess.Popen(['../../bin/borderless', s], stdout=subprocess.PIPE)
    #         _, borderless_length, borderless = process.stdout.read().strip().split()
    #         if int(borderless_length) < length:
    #             printed += 1
    #             print str(length).ljust(2), s.ljust(MAX_LEN),
    #             print borderless_length.ljust(2), borderless.ljust(MAX_LEN), period
    # print 'Total generated words count:', generated, 'selected non-borderless word count:', EXPERIMENTS, '({:.2f})'.format(EXPERIMENTS * 1.0 / generated)
    for size in xrange(2, 3):
        p = [0, 0]
        for n in xrange(2, 30):
            q = [0] * (n + 1)
            words = [[] for i in xrange(n + 1)]
            for word in all_words(AVAILABLE_LETTERS[:size], n):
                process = subprocess.Popen(['../../bin/border', word], stdout=subprocess.PIPE)
                border_length = int(process.stdout.read().strip())
                q[n - border_length] += 1
                words[n - border_length].append(word)
            print 'sigma =', size, 'n =', n
            for i, q_i in enumerate(q[1:]):
                i += 1
                print 'i =', i, 'q_i =', q_i, 'est =', size ** i - sum(size ** j for j in [1, 2, 3, 5, 7, 11] if j < i and i % j == 0), ':'
                # print '\n'.join(words[i])
            print
            # answer = size * p[n - 1]
            # if n % 2 == 0:
            #     answer += size ** (n / 2) - p[n / 2]
            # p.append(answer)
            # print non_borderless, answer
            # print 'n = {}'.format(n).ljust(6), ' P(sigma, n) = {}'.format(answer).ljust(24), '({:.3f} sigma^n)'.format(answer * 1.0 / size ** n)
