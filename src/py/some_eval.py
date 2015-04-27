# -*- coding: utf-8 -*-
import sys

DEFAULT_MAX_VALUE = 100


def f(x):
    return (2 * (x ** 4) - 3 * (x ** 3) + 2 * (x ** 2)) * 1.0 / ((x - 1) ** 2) / (x ** 2 - 2 * x + 2)


if __name__ == '__main__':
    max_value = DEFAULT_MAX_VALUE
    if len(sys.argv) > 1:
        try:
            max_value = int(sys.argv[1])
        except:
            pass

    for i in xrange(2, max_value):
        print str(i).rjust(3), str('{:.5f}'.format(f(i))).rjust(10)
