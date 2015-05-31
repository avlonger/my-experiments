# -*- coding: utf-8 -*-
"""
"""
import itertools
from collections import namedtuple

import pylab as pl
import matplotlib.font_manager as fm


DASHES = [
    [],
    [5, 5],
    [5, 3, 1, 3],
    [1, 1],
]

FONT_PATH = '/Users/alonger/HSE/cmunrm.ttf'

Line = namedtuple('Line', ['x', 'y', 'label'])


def art_plot(lines, filename, black_and_white=True, clear_before_plotting=True, font_size=12, title=None,
             legend_location=None, xlabel=None, ylabel=None, xlim=None, ylim=None, mirror_ymin=False):
    font = fm.FontProperties(fname=FONT_PATH, size=font_size)
    if clear_before_plotting:
        pl.axes().clear()
    for line, dashes in zip(lines, itertools.cycle(DASHES)):
        if black_and_white:
            pl.plot(line.x, line.y, color='k', dashes=dashes, label=line.label)
        else:
            pl.plot(line.x, line.y, label=line.label)

    if xlim is not None:
        pl.axes().set_xlim(xlim)

    if mirror_ymin and ylim is None:
        ymin, ymax = pl.axes().get_ylim()
        ymin = -ymax * 0.25
        pl.axes().set_ylim((ymin, ymax))

    if ylim is not None:
        pl.axes().set_ylim(ylim)

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
