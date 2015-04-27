# -*- coding: utf-8 -*-
"""
"""
import csv
import time
import argparse
import datetime
import subprocess
import multiprocessing as mp
from multiprocessing.pool import ThreadPool


def worker(args):

    alphabet, length, step, diff, seed, algo, binary = args
    start = time.time()
    p = subprocess.Popen([
        binary, '-a', str(alphabet), '-b', str(length), '-f', str(length + 1), '-s', str(step), '-d', str(diff), algo,
        '-r', str(seed)
    ], stdout=subprocess.PIPE)
    p.wait()
    return p.stdout.read().strip().split() + [time.time() - start]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate some statistics')
    parser.add_argument('-a', '--alphabets', nargs='*', type=int,
                        help='Alphabet sizes for which calculate statistics')
    parser.add_argument('-l', '--lengths', nargs='*', type=int,
                        help='Word lengths for which calculate statistics')
    parser.add_argument('-c', '--cpu_count', type=int,
                        help='Processor cores to use (default: all available)',
                        default=mp.cpu_count())
    parser.add_argument('-f', '--filename', help='Output file')
    parser.add_argument('-b', '--binary', help='Binary to use', default='../../bin/experiment')
    parser.add_argument('-s', '--step', help='Step', default=10 ** 6, type=int)
    parser.add_argument('-d', '--difference', help='Target tolerance', default=0.00000001, type=float)
    parser.add_argument('-r', '--seed', help='Random seed', default=1961, type=int)
    parser.add_argument('algo', metavar='ALGORITHM',
                        choices=('MAXBORDERLESS_SUBWORD', 'MAXBORDERLESS_PREFIX'))
    options = parser.parse_args()

    pool = ThreadPool(options.cpu_count)

    if options.filename is None:
        options.filename = 'ESTIMATION-{}-{}.txt'.format(options.algo, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))

    with open(options.filename, 'w') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerow(('ALPHABET', 'LENGTH', 'SUM', options.algo, 'TIME'))
        tasks = []

        for alphabet_size in options.alphabets:
            for length in options.lengths:
                tasks.append((
                    alphabet_size, length, options.step, options.difference, options.seed, options.algo, options.binary
                ))

        for result in pool.imap_unordered(worker, tasks):
            writer.writerow(result)
            out_file.flush()
