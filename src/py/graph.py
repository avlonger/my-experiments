from __future__ import division
import os
from collections import defaultdict, namedtuple

import pylab as pl
import matplotlib.font_manager as fm


def save_me(name, font, clear=True):
    set_font(font)
    pl.tight_layout()
    pl.savefig(name)
    if clear:
        pl.axes().clear()


def set_font(font):
    pl.xlabel('String length', fontproperties=font)
    for label in pl.axes().get_xticklabels():
        label.set_fontproperties(font)
    for label in pl.axes().get_yticklabels():
        label.set_fontproperties(font)

dashes = [
    [],
    [5, 5],
    [5, 3, 1, 3],
    [1, 1],
]

MAX_COMPOSITE_LENGTH = 100

Entity = namedtuple('Entity', ['name', 'readable_name', 'symbol'])

if __name__ == '__main__':

    font19 = fm.FontProperties(fname='/Users/alonger/HSE/cmunrm.ttf', size=19)
    font14 = fm.FontProperties(fname='/Users/alonger/HSE/cmunrm.ttf', size=14)

    entities = [
        Entity('borderless', 'unbordered factor', r'\beta(n)'),
        Entity('borderless_prefix', 'unbordered prefix', r'\pi(n)'),
    ]

    for name, verbose_name, symbol in entities:

        counts = defaultdict(lambda: defaultdict(int))

        with open('max_{}.txt'.format(name)) as fd:
            for line in fd:
                alphabet, length, val = map(int, line.strip().split())
                counts[alphabet][length] = length - val / alphabet ** length

        with open('max_{}_estimation.txt'.format(name)) as fd:
            for line in fd:
                alphabet, length, _, _, val = line.strip().split()[:5]
                counts[int(alphabet)][int(length)] = int(length) - float(val)

        dir_name = 'Average difference/Between n and maximal {}'.format(verbose_name)

        for alphabet in xrange(2, 6):
            keys = sorted(counts[alphabet])
            values = [counts[alphabet][x] for x in keys]
            pl.plot(keys, values, color='k')
            pl.axes().set_xlim(min(keys), max(keys))
            try:
                os.mkdir('results/{}'.format(dir_name))
                os.mkdir('for_paper/{}'.format(dir_name))
            except Exception:
                pass
            save_me('for_paper/{}/Alphabet_size_{}.pdf'.format(dir_name, alphabet), font19)

            pl.plot(keys, values)
            pl.axes().set_xlim(min(keys), max(keys))
            pl.grid()

            pl.title('Average difference between the length $n$ of a string and the length\n'
                     'of its maximal {} for alphabet of size $\sigma = {}$\n'.format(verbose_name, alphabet),
                     fontproperties=font14)
            save_me('results/{}/Alphabet_size_{}.png'.format(dir_name, alphabet), font14)

        for alphabets in [range(2, 6), range(3, 6)]:
            for i, alphabet in enumerate(alphabets):
                keys = sorted(counts[alphabet])[:MAX_COMPOSITE_LENGTH]
                values = [counts[alphabet][x] for x in keys]
                pl.plot(keys, values, label='$\sigma = {}$'.format(alphabet), color='k', dashes=dashes[i])
            pl.axes().set_xlim(min(keys), max(keys))
            pl.legend(loc=4, prop=font19)
            save_me('for_paper/{}/Alphabet_size_{}.pdf'.format(dir_name, '_'.join(map(str, alphabets))), font19)

            for i, alphabet in enumerate(alphabets):
                keys = sorted(counts[alphabet])[:MAX_COMPOSITE_LENGTH]
                values = [counts[alphabet][x] for x in keys]
                pl.plot(keys, values, label='$\sigma = {}$'.format(alphabet))
            pl.axes().set_xlim(min(keys), max(keys))
            pl.legend(loc=4, prop=font14)
            pl.grid()

            pl.title('Average difference between the length $n$ of a string and the length\n'
                     'of its maximal {} for alphabets of size $\sigma = \{{{}\}}$\n'.format(
                         verbose_name, ','.join(map(str, alphabets))), fontproperties=font14)
            save_me('results/{}/Alphabet_size_{}.png'.format(dir_name, '_'.join(map(str, alphabets))), font14)

    for alphabet in xrange(2, 6):

        if alphabet == 2:
            loc = 2
        else:
            loc = 4

        dir_name = 'Average difference/Between n and maximal unbordered factor and prefix'

        try:
            os.mkdir('results/{}'.format(dir_name))
            os.mkdir('for_paper/{}'.format(dir_name))
        except Exception:
            pass

        for i, (name, verbose_name, symbol) in enumerate(entities):

            counts = defaultdict(lambda: defaultdict(int))

            with open('max_{}.txt'.format(name)) as fd:
                for line in fd:
                    alphabet_size, length, val = map(int, line.strip().split())
                    counts[alphabet_size][length] = length - val / alphabet_size ** length

            with open('max_{}_estimation.txt'.format(name)) as fd:
                for line in fd:
                    alphabet_size, length, _, _, val = line.strip().split()[:5]
                    counts[int(alphabet_size)][int(length)] = int(length) - float(val)

            keys = sorted(counts[alphabet])
            values = [counts[alphabet][x] for x in keys]
            pl.plot(keys, values, color='k', dashes=dashes[i], label='$n - {}$'.format(symbol))

        pl.legend(loc=loc, prop=font14)
        pl.axes().set_xlim(2, 100)
        save_me('for_paper/{}/Alphabet_size_{}.pdf'.format(dir_name, alphabet), font19)

        for i, (name, verbose_name, symbol) in enumerate(entities):

            counts = defaultdict(lambda: defaultdict(int))

            with open('max_{}.txt'.format(name)) as fd:
                for line in fd:
                    alphabet_size, length, val = map(int, line.strip().split())
                    counts[alphabet_size][length] = length - val / alphabet_size ** length

            with open('max_{}_estimation.txt'.format(name)) as fd:
                for line in fd:
                    alphabet_size, length, _, _, val = line.strip().split()[:5]
                    counts[int(alphabet_size)][int(length)] = int(length) - float(val)

            keys = sorted(counts[alphabet])[:100]
            values = [counts[alphabet][x] for x in keys][:100]
            pl.plot(keys, values, label='$n - {}$'.format(symbol))

        pl.legend(loc=loc, prop=font14)
        pl.grid()
        pl.axes().set_xlim(2, 100)
        # pl.title('Difference between the length $n$ of a string and the average length of its maximal\n'
        #          'unbordered factor $\\beta(n)$ and maximal unbordered prefix $\\pi(n)$ '
        #          'for alphabet of size $\sigma={}$'.format(alphabet), fontproperties=font14)
        save_me('results/{}/Alphabet_size_{}.png'.format(dir_name, alphabet), font14, dpi=900)

        pl.axes().clear()
