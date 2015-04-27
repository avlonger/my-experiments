# -*- coding: utf-8 -*-
from __future__ import division
from collections import defaultdict


TEMPLATE = '''
##### The average difference between the length of a string and the length of its maximal unbordered {entity} length

{diff_table}

##### The average length of the maximal unbordered {entity} length of a string

{value_table}
'''


def justify_numbers(numbers, width=7):
    result = ''
    for number in numbers:
        if isinstance(number, int):
            result += str(number).rjust(width)
        else:
            result += '{:.3f}'.format(number).rjust(width)
    return result


if __name__ == '__main__':
    entity = 'prefix'
    values = defaultdict(lambda: defaultdict(int))

    max_binary_value = 0
    max_non_binary_value = 0

    with open('max_borderless_{}.txt'.format(entity)) as fd:
        for line in fd:
            alphabet, length, value = [int(i) for i in line.strip().split()]
            values[alphabet][length] = value
            if alphabet == 2 and length > max_binary_value:
                max_binary_value = length
            elif alphabet != 2 and length > max_non_binary_value:
                max_non_binary_value = length

    for alphabets, length in [([2], max_binary_value), (values.keys(), max_non_binary_value)]:
        table = '     | ' + ' | '.join(str(i) for i in xrange(2, length + 1)) + '\n'
        table += ' --: | ' + ' | '.join('--:' for i in xrange(2, length + 1)) + '\n'
        diff_table = value_table = table

        for alphabet in alphabets:
            results = [values[alphabet][i] / alphabet ** i for i in xrange(2, length + 1)]
            diff_results = [i - values[alphabet][i] / alphabet ** i for i in xrange(2, length + 1)]
            value_table += '{} | '.format(alphabet) + ' | '.join('{:.3f}'.format(i) for i in results) + '\n'
            diff_table += '{} | '.format(alphabet) + ' | '.join('{:.3f}'.format(i) for i in diff_results) + '\n'

        with open('max_borderless_{}_{}.md'.format(entity, alphabets[-1]), 'w') as fd:
            fd.write(TEMPLATE.format(entity=entity, diff_table=diff_table, value_table=value_table))


