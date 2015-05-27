# -*- coding: utf-8 -*-
import csv
import time
import argparse
import datetime
import subprocess
import multiprocessing as mp
from multiprocessing.pool import ThreadPool


def worker(args):
    size, length, algo, binary, prefix, prefix_length = args
    start = time.time()
    p = subprocess.Popen(
        [
            binary, '-a', str(size), '-b', str(length), '-e', str(length + 1), '-s', '1',
            '-f', str(prefix_length), '-p', str(prefix), algo
        ],
        stdout=subprocess.PIPE
    )
    p.wait()
    return size, length, long(p.stdout.read().strip().split()[1]), time.time() - start

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
    parser.add_argument('algo', metavar='ALGORITHM',
                        choices=('MAXBORDERLESS_BORDER', 'MAXBORDERLESS_PREFIX'))
    options = parser.parse_args()

    pool = ThreadPool(options.cpu_count)

    if options.filename is None:
        options.filename = '{}-{}.txt'.format(options.algo, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))

    with open(options.filename, 'w') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerow(('ALPHABET', 'LENGTH', 'SUM', options.algo, 'TIME'))
        for alphabet_size in options.alphabets:

            tasks_count = 1
            # because we can multiply the answer for A...
            # by the alphabet size, cause the same numbers
            # would be obtained for B..., C..., etc.
            fixed_prefix_length = 1
            while tasks_count < options.cpu_count:
                tasks_count *= alphabet_size
                fixed_prefix_length += 1

            for length in options.lengths:
                if options.cpu_count < alphabet_size ** length and length > 9:
                    tasks = [
                        (alphabet_size, length, options.algo,
                         options.binary, i, fixed_prefix_length) for i in xrange(tasks_count)
                    ]
                else:
                    tasks = [(alphabet_size, length, options.algo, options.binary, 0, 0)]

                for size, length, result, spent_time in pool.imap_unordered(worker, tasks):

                    # because we have calculated the answer for A... words
                    result *= alphabet_size

                    writer.writerow((size, length, result, '{:.2f}'.format(spent_time)))
                    out_file.flush()
