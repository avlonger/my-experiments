# -*- coding: utf-8 -*-
"""
Module for experiments on values of b_j and v_j
"""
from __future__ import division

import time
import string
import datetime
import itertools
from collections import defaultdict, namedtuple

import pylab as pl
import matplotlib.font_manager as fm


def all_words(available_letters, word_length):
    """
    Generate all distinct words for given alphabet and word length
    """
    for result_word in itertools.product(available_letters, repeat=word_length):
        yield ''.join(result_word)


# dashes for plotting in black and white
DASHES = [
    [],
    [5, 5],
    [5, 3, 1, 3],
    [1, 1],
    ]


# class for line for plotting representation
Line = namedtuple('Line', ['x', 'y', 'label'])


def art_plot(lines, filename, black_and_white=True, clear_before_plotting=True, font_size=12, title=None,
             legend_location=None, xlabel=None, ylabel=None, xlim=None, mirror_ymin=False):
    """
    Some helper for plotting
    """
    font = fm.FontProperties(fname='cmunrm.ttf', size=font_size)
    if clear_before_plotting:
        pl.axes().clear()
    for line, dashes in zip(lines, itertools.cycle(DASHES)):
        if black_and_white:
            pl.plot(line.x, line.y, color='k', dashes=dashes, label=line.label)
        else:
            pl.plot(line.x, line.y, label=line.label)

    if xlim is not None:
        pl.axes().set_xlim(xlim)

    if mirror_ymin:
        ymin, ymax = pl.axes().get_ylim()
        ymin = -ymax * 0.25
        pl.axes().set_ylim((ymin, ymax))

    if legend_location is not None:
        pl.legend(loc=legend_location, prop=font)

    if title is not None:
        pl.title(title, fontproperties=font)

    if xlabel is not None:
        pl.xlabel(xlabel, fontproperties=font)

    if ylabel is not None:
        pl.ylabel(ylabel, fontproperties=font)

    for label in pl.axes().get_xticklabels():
        label.set_fontproperties(font)

    for label in pl.axes().get_yticklabels():
        label.set_fontproperties(font)

    pl.tight_layout()
    pl.savefig(filename)


def border(text):
    """
    Builds border array for given text
    """
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


def is_unbordered(text):
    """
    Test if given word is unbordered
    """
    return border(text)[-1] == 0


def unequal_count_after_first_letter(word):
    """
    Find how many consequent letters is unequal to the first letter of a word
    """
    for i in xrange(1, len(word)):
        if word[i] == word[0]:
            return i - 1
    return len(word) - 1


if __name__ == '__main__':
    max_lens = {
        2: 21,
        # 3: 21,
        # 5: 21,
        # 8: 21,
    }

    for alphabet, max_len in max_lens.iteritems():

        result = defaultdict(lambda: defaultdict(int))

        # calculate b_j
        for length in xrange(1, max_len):
            start = time.time()
            print alphabet, length, datetime.datetime.now(),
            for word in all_words(string.uppercase[:alphabet], length):
                if word[0] != 'A':
                    break
                if is_unbordered(word):
                    j = unequal_count_after_first_letter(word)
                    for l in xrange(j + 1):
                        result[l][length] += 1
            print int(time.time() - start), 'sec'

        # plot v_j
        for j, values_per_length in result.iteritems():
            q = alphabet
            for length in sorted(values_per_length):
                if j == 0:
                    continue
                my_value = q ** (length - j) * (q - 1) ** (j - 1) * (q - 2)
                assert my_value < values_per_length[length] * q

            lengths = sorted(values_per_length)
            lines = [
                Line(
                    lengths,
                    [(q - 1) ** (j - 1) * (q - 2) / q ** j] * len(lengths),
                    '$\sigma^{-j}(\sigma - 1)^{j - 1}(\sigma - 2)$',
                ),
                Line(
                    lengths,
                    [values_per_length.get(l) * q / q ** l for l in lengths],
                    '$v_n$',
                )
            ]
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
