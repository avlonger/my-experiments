# -*- coding: utf-8 -*-
"""
"""
import pylab as pl
import matplotlib.font_manager as fm


def set_font(font):
    pl.xlabel('String length', fontproperties=font)
    pl.tight_layout()
    for label in pl.axes().get_xticklabels():
        label.set_fontproperties(font)
    for label in pl.axes().get_yticklabels():
        label.set_fontproperties(font)


def save_me(name, font):
    set_font(font)
    pl.savefig('final_figs/' + name)
    pl.savefig('final_figs/' + name[:-4] + '.png')
    pl.axes().clear()


if __name__ == '__main__':

    font = fm.FontProperties(fname='/Users/alonger/HSE/cmunrm.ttf', size=14)

    for a in [2, 3, 4, 5, 10]:
        with open('comparison{}.txt'.format(a)) as fd:
            lines = fd.read().strip().splitlines()
            super_naive = [(l.split()[1], float(l.split()[2]) * 1000) for l in lines if 'BASIC' in l]
            border = [(l.split()[1], float(l.split()[2]) * 1000) for l in lines if 'PROPOSED' in l]
            dbf = [(l.split()[1], float(l.split()[2]) * 1000) for l in lines if 'DBF_ALGORITHM' in l]
            dbf_hash = [(l.split()[1], float(l.split()[2]) * 1000) for l in lines if 'HASH' in l]

        pl.plot(*zip(*border), label='Proposed algorithm')
        pl.plot(*zip(*super_naive), label='Basic algorithm')
        pl.plot(*zip(*dbf), label='DBF')
        pl.plot(*zip(*dbf_hash), label='DBF + HashTable')

        pl.ylabel('Time, $10^{-3}$s', fontproperties=font)
        pl.title('Average running time of the proposed algorithm, the basic algorithm\n'
                 'and the algorithms based on Dictionary of Basic Factors,\n'
                 'for the alphabet of size $\sigma = {}$\n'.format(a), fontproperties=font)
        pl.legend(loc=2, prop=font)
        save_me('comparison{}.png'.format(a), font)
