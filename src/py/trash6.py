# -*- coding: utf-8 -*-
"""
"""
from __future__ import division
import math


def fib(n):
    return (((1 + math.sqrt(5)) / 2) ** n - ((1 - math.sqrt(5)) / 2) ** n) / math.sqrt(5)


def fib_bound(n):
    return 1.63 ** (n - 2)


if __name__ == '__main__':
    for n in xrange(1, 200):
        print fib(n), fib_bound(n)
        assert fib(n) >= fib_bound(n)
