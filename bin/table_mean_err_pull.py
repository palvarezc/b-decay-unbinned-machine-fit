#!/usr/bin/env python
"""Output table of signal values, means, std errs and pulls"""
import argparse
import csv
import numpy as np
import scipy.stats
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Output table of signal values, means, std errs and pulls.',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=columns, width=columns),
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

print('\\begin{table}[h!]')
print('\\centering')
print(' \\begin{tabular}{|c|c|c|c|c|}')
print(' \\hline')
print(' Coefficient & Signal & Mean & Std Err & Pull Mean \\\\')
print(' \\hline')

for c_name, signal_value in signal_coeffs.items():
    c_id = bmf.coeffs.names.index(c_name)
    if c_id not in bmf.coeffs.fit_trainable_idxs:
        continue

    mean = np.mean(data_points[c_name])
    std_err = scipy.stats.sem(data_points[c_name], axis=None)
    pull = list(map(lambda p: (p - signal_value) / std_err, data_points[c_name]))
    pull_mean = np.mean(pull)

    print(
        ' {} & {:+.5f} & {:+.5f} & {:+.5f} & {:+.5f} \\\\'.format(
            bmf.coeffs.latex_names[c_id],
            signal_value,
            mean,
            std_err,
            pull_mean
        )
    )

print(' \\hline')
print(' \\end{tabular}')
print('\\caption{Write me}')
print('\\label{table:mean_err_pull}')
print('\\end{table}')
