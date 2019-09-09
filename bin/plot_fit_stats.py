#!/usr/bin/env python
"""
Plot two stats against each other for fit CSVs
"""
import argparse
import csv
import matplotlib
import numpy as np
import os
import scipy.stats
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()


def filename_and_name(arg):
    try:
        _filename, _name = arg.split(":")
    except ValueError:
        _filename = arg
        _name = os.path.splitext(os.path.basename(_filename))[0]
    return _filename, _name


axes = {
    'signal': 'Signal',
    'abs_signal': 'Absolute Signal',
    'diff': 'Difference',
    'std_err': 'Standard Error',
    'pull_mean': 'Pull Mean'
}

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot two stats against each other for fit CSVs.',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=columns, width=columns),
)
parser.add_argument(
    '-d',
    '--device',
    dest='device',
    default=bmf.Script.device_default,
    help='use this device e.g. CPU:0, GPU:0, GPU:1 (default: {})'.format(bmf.Script.device_default),
)
parser.add_argument(
    '-w',
    '--write-svg',
    dest='write_svg',
    metavar='SVG_PATH',
    help='write plots as SVGs using this filepath'
)
parser.add_argument(
    '-x',
    '--x-axis',
    dest='x_axis',
    choices=axes.keys(),
    required=True,
    help='what to plot on the x-axis',
)
parser.add_argument(
    '-y',
    '--y-axis',
    dest='y_axis',
    choices=axes.keys(),
    required=True,
    help='what to plot on the y-axis',
)
parser.add_argument(
    nargs='+',
    dest='plot_list',
    action='append',
    type=filename_and_name,
    metavar='FILENAME[:NAME]',
    help='Filename and optional name to plot (e.g. NP_0.15.csv or NP_0.15.csv:"NP 0.15")'
)
args = parser.parse_args()

with bmf.Script(device=args.device) as script:
    if args.write_svg is not None:
        matplotlib.use('SVG')

    # Import these after we optionally set SVG backend - otherwise matplotlib may bail on a missing TK backend when
    #  running from the CLI
    import matplotlib.pylab as plt
    import seaborn as sns

    # Load input
    data_points = {}
    signal_coeffs = {}
    for plot in args.plot_list[0]:
        filename, p_name = plot
        with open(filename, newline='') as csv_file:
            reader = csv.DictReader(csv_file)

            # Load signal
            if p_name not in signal_coeffs:
                signal_coeffs[p_name] = {}
            signal = next(reader)
            for c_name in bmf.coeffs.names:
                signal_coeffs[p_name][c_name] = float(signal[c_name])

            # Load fit coefficients
            for row in reader:
                for c_name in bmf.coeffs.names:
                    if row[c_name] == "0.0":
                        continue
                    if c_name not in data_points:
                        data_points[c_name] = {}
                    if p_name not in data_points[c_name]:
                        data_points[c_name][p_name] = []
                    data_points[c_name][p_name].append(float(row[c_name]))

    # Calculate what to plot
    plot_list = {}
    for name in axes.keys():
        plot_list[name] = {}
    for p_name in signal_coeffs.keys():
        for c_name, signal_value in signal_coeffs[p_name].items():
            c_id = bmf.coeffs.names.index(c_name)
            if c_id not in bmf.coeffs.fit_trainable_idxs:
                continue

            mean = np.mean(data_points[c_name][p_name])
            diff = mean - signal_value
            std_err = scipy.stats.sem(data_points[c_name][p_name], axis=None)
            pull = list(map(lambda p: (p - signal_value) / std_err, data_points[c_name][p_name]))
            pull_mean = np.mean(pull)

            if p_name not in plot_list['signal']:
                plot_list['signal'][p_name] = []
            plot_list['signal'][p_name].append(signal_value)

            if p_name not in plot_list['abs_signal']:
                plot_list['abs_signal'][p_name] = []
            plot_list['abs_signal'][p_name].append(abs(signal_value))

            if p_name not in plot_list['diff']:
                plot_list['diff'][p_name] = []
            plot_list['diff'][p_name].append(diff)

            if p_name not in plot_list['std_err']:
                plot_list['std_err'][p_name] = []
            plot_list['std_err'][p_name].append(std_err)

            if p_name not in plot_list['pull_mean']:
                plot_list['pull_mean'][p_name] = []
            plot_list['pull_mean'][p_name].append(pull_mean)

    # Do plots
    plt.figure()
    # Set style as well as font to Computer Modern Roman to match LaTeX output
    sns.set(style='ticks', font='cmr10', rc={'mathtext.fontset': 'cm', 'axes.unicode_minus': False})

    for p_name in signal_coeffs.keys():
        plt.scatter(plot_list[args.x_axis][p_name], plot_list[args.y_axis][p_name], label=p_name, marker='x')

    plt.xlabel(axes[args.x_axis])
    plt.ylabel(axes[args.y_axis])
    plt.margins(x=0)

    if len(plot_list) > 1:
        plt.legend()

    if args.write_svg is not None:
        filepath = args.write_svg
        bmf.stdout('Writing {}'.format(filepath))
        plt.savefig(filepath, format='svg', bbox_inches='tight')
    else:
        plt.show()
