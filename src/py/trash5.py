# -*- coding: utf-8 -*-
"""
"""
from __future__ import division
import math
from collections import defaultdict

import pylab as pl


def b(j, i, sigma):
    if j == 0:
        return sigma ** i * (1 - 1 / sigma - 1 / sigma / sigma)
    if j > i - 1:
        return 0
    return sigma ** (i - j) * (sigma - 1) ** (j - 1) * (sigma - 2)


def eta(i, j, sigma, n):
    if n - i <= j + 1:
        return 2 ** (n - i - 1)
    # FIXME: want some better
    return fib_bound(n - i + 1)


fib_table = defaultdict(lambda: defaultdict(int))


def fib_order(n, order):
    if n < 1:
        return 0
    if n == 1:
        return 1
    if not fib_table[order][n]:
        fib_table[order][n] = sum(fib_order(n - i, order) for i in xrange(1, order + 1))
    return fib_table[order][n]


def fib(n):
    return (((1 + math.sqrt(5)) / 2) ** n - ((1 - math.sqrt(5)) / 2) ** n) / math.sqrt(5)


def fib_bound(n):
    return 1.63 ** (n - 2)


def a_(j):
    return (5 + 3 * math.sqrt(5)) / 10 * 2 ** (j - 1)


def b_(j):
    return (5 - 3 * math.sqrt(5)) / 10 * 2 ** (j - 1)


phi = (1 + math.sqrt(5)) / 2


ksi = (1 - math.sqrt(5)) / 2


def my_f_2(i, j, n):
    if j == 0:
        return 1
    return phi ** (n - i - j) * (5 + 3.0 * math.sqrt(5)) / 10 * 2 ** (j - 1) + ksi ** (n - i - j) * b_(j)


def myf(i, j, n):
    return fib(n - i - j) * 2 ** (j - 2)


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


c_table = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def c(j, k, i, sigma):
    if k == 1:
        return b_precise(j, i, sigma)
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


def new(n, sigma):
    result = 0

    for i in xrange(1, n + 1):
        result += i * b_0_precise(i, sigma)

    for i in xrange(1, n - 1):
        result += i * (fib(n - i + 2) / 2 - 1) * b_precise(1, i, sigma)

    for i in xrange(1, n - 1):
        sub_result = 0
        for j in xrange(1, n - i):
            sub_result += myf(i, j, n) * b_precise(j, i, sigma)

        result += i * sub_result

    return result / sigma ** n


skip_fib_table = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def skip_fib_order(n, order, skip):
    if n < 1:
        return 0
    if n == 1:
        return 1
    if order == 1:
        return 1
    if not skip_fib_table[order][skip][n]:
        skip_fib_table[order][skip][n] = skip_fib_order(n - 1, order, skip) + sum(skip_fib_order(n - i, order, skip) for i in xrange(skip + 2, skip + 1 + order))
    return skip_fib_table[order][skip][n]


def test(n, sigma):
    result = 0

    for i in xrange(1, n + 1):
        result += i * b_0_precise(i, sigma)

    for i in xrange(1, n - 1):
        sub_result = 0
        for k in xrange(1, i):
            for j in xrange(1, i - k + 1):
                sub_result += (skip_fib_order(n - i + 1, j + 1, k - 1) - skip_fib_order(n - i + 1, j, k - 1)) * c(j, k, i, sigma)
        result += i * sub_result

    return result / sigma ** n


def old(n, sigma):
    result = 0

    for i in xrange(1, n + 1):
        result += i * b(0, i, sigma)

    for i in xrange(n // 2, n):
        sub_result = 0
        for j in xrange(1, n - i):
            sub_result += 2 ** (j - 1) * b(j, i, sigma)

        result += i * sub_result

    return result / sigma ** n

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


beta_table = defaultdict(lambda: defaultdict(int))


def beta(i, n, sigma):
    if i == n:
        return b_0_precise(i, sigma)
    if i > n:
        return 0
    if beta_table[i][n] == 0:
        result = b_0_precise(i, sigma) * (beta(i, n - 1, sigma) + beta(i, n - i, sigma))
        for j in xrange(2, i):
            result += delta(i, j, sigma) * beta(i, n - j, sigma)
        beta_table[i][n] = result
    return beta_table[i][n]


def ultra_new(n, sigma):
    return sum(i * beta(i, n, sigma) for i in xrange(1, n + 1)) / sigma ** n


if __name__ == '__main__':
    size = 4
    for n in xrange(2, 101):
        print n, ultra_new(n, size), new(n, size), old(n, size)
    # pl.plot(range(2, 101), result)
    # pl.savefig('alphabet{}.png'.format(size))

