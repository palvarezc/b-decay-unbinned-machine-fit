#!/usr/bin/env python
"""Output table of signal values, means, std errs and pulls"""
import argparse
import csv
import numpy as np
import scipy.stats
import shutil

import os
# Disable non-important TF log lines
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import b_meson_fit as bmf


def make_slice(expr):
    def to_piece(s):
        return s and int(s) or None
    pieces = list(map(to_piece, expr.split(':')))
    if len(pieces) == 1:
        return slice(pieces[0], pieces[0] + 1)
    else:
        return slice(*pieces)


def make_columns(arg):
    columns = arg.split(',')
    allowed_columns = column_list.keys()
    for column in columns:
        if column not in allowed_columns:
            raise argparse.ArgumentTypeError(
                '{} is not one of {}'.format(column, ",".join(allowed_columns))
            )
    return columns


column_list = {
    'signal': 'Signal',
    'mean': 'Mean',
    'diff': 'Difference',
    'std_err': 'Std. Err',
    'pull_mean': 'Pull Mean'
}

term_columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Output table of signal values, means, std errs and pulls.',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=term_columns, width=term_columns),
)
parser.add_argument(
    '-c',
    '--slice',
    dest='slice',
    type=make_slice,
    help='slice the results using this string (e.g. \'0:3\'  or \' -3:\' [note the space])'
)
parser.add_argument(
    '-l',
    '--columns',
    dest='columns',
    type=make_columns,
    default=column_list.keys(),
    help='display these columns (default: {}'.format(','.join(column_list.keys()))
)
parser.add_argument(
    '-s',
    '--sort',
    dest='sort',
    choices=column_list.keys(),
    help='sort by this key'
)
parser.add_argument(
    dest='filename',
    metavar='FILENAME',
    help='Path to fit CSV file'
)

args = parser.parse_args()

# Load input
data_points = {}
signal_coeffs = {}
with open(args.filename, newline='') as csv_file:
    reader = csv.DictReader(csv_file)

    # Load signal
    signal = next(reader)
    for c_name in bmf.coeffs.names:
        signal_coeffs[c_name] = float(signal[c_name])

    # Load fit coefficients
    for row in reader:
        for c_name in bmf.coeffs.names:
            if c_name not in data_points:
                data_points[c_name] = []
            data_points[c_name].append(float(row[c_name]))

c_list = []
for c_name, signal_value in signal_coeffs.items():
    c_id = bmf.coeffs.names.index(c_name)
    if c_id not in bmf.coeffs.fit_trainable_idxs:
        continue

    mean = np.mean(data_points[c_name])
    std_err = scipy.stats.sem(data_points[c_name], axis=None)
    pull = list(map(lambda p: (p - signal_value) / std_err, data_points[c_name]))
    pull_mean = np.mean(pull)

    c_list.append({
        'name': bmf.coeffs.latex_names[c_id],
        'signal': signal_value,
        'mean': mean,
        'diff': mean - signal_value,
        'std_err': std_err,
        'pull_mean': pull_mean
    })

if args.sort is not None:
    c_list = sorted(c_list, key=lambda i: abs(i[args.sort]), reverse=True)

if args.slice is not None:
    c_list = c_list[args.slice]

print('\\begin{table}[h!]')
print('\\centering')
print(' \\begin{{tabular}}{{|{}|}}'.format('|'.join(['c'] * (len(args.columns) + 1))))
print(' \\hline')
print(' Coefficient', end='')
for name, display_name in column_list.items():
    if name in args.columns:
        print(' & {}'.format(display_name), end='')
print(' \\\\')
print(' \\hline')

for coeff in c_list:
    line = ' {}'.format(coeff['name'])
    if 'signal' in args.columns:
        line = line + ' & {:+.5f}'.format(coeff['signal'])
    if 'mean' in args.columns:
        line = line + ' & {:+.5f}'.format(coeff['mean'])
    if 'diff' in args.columns:
        line = line + ' & {:+.5f}'.format(coeff['diff'])
    if 'std_err' in args.columns:
        line = line + ' & {:.5f}'.format(coeff['std_err'])
    if 'pull_mean' in args.columns:
        line = line + ' & {:+.5f}'.format(coeff['pull_mean'])
    line = line + ' \\\\'
    print(line)

print(' \\hline')
print(' \\end{tabular}')
print('\\caption{Write me}')
print('\\label{table:mean_err_pull}')
print('\\end{table}')
