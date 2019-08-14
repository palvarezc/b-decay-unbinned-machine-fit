#!/usr/bin/env python
"""
Plot coefficient value and pull histograms for given CSV result files.

Will also output mean, std err and pull mean for each coefficient.
"""

import argparse
import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import seaborn as sns
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


columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Fit coefficients to generated toy signal(s).',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=columns, width=columns),
)
parser.add_argument(
    '-b',
    '--bins',
    dest='bins',
    type=int,
    default=100,
    help='Number of histogram bins (default: 100)',
)
parser.add_argument(
    '-s',
    '--plot-scale',
    dest='plot_scale',
    type=float,
    default=12.0,
    help='Plot from +/- of this value (default: 12.0)',
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

with bmf.Script(device=None) as script:
    # Load inputs
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
                    if c_name not in data_points:
                        data_points[c_name] = {}
                    if p_name not in data_points[c_name]:
                        data_points[c_name][p_name] = []
                    data_points[c_name][p_name].append(float(row[c_name]))

    # For each amplitude
    for a_idx in range(0, bmf.coeffs.amplitude_count):
        fig, axes = plt.subplots(bmf.coeffs.param_count * 2)
        fig.suptitle(bmf.coeffs.amplitude_latex_names[a_idx])

        # For each param in this amplitude
        for p_idx in range(0, bmf.coeffs.param_count):
            c_idx = a_idx * bmf.coeffs.param_count + p_idx
            c_name = bmf.coeffs.names[c_idx]
            colors = itertools.cycle(sns.color_palette("Set2"))

            drawn_something = False
            for name, points in data_points[c_name].items():
                if not all(elem == 0.0 for elem in points):
                    mean = np.mean(points)
                    std_err = sp.stats.sem(points, axis=None)
                    pull = list(map(lambda p: (p - signal_coeffs[name][c_name]) / std_err, points))
                    pull_mean = np.mean(points)

                    bmf.stdout(
                        '{}/{} signal: {} mean: {} std err: {} pull mean: {}'.format(
                            bmf.coeffs.names[c_idx],
                            name,
                            signal_coeffs[name][c_name],
                            mean,
                            std_err,
                            pull_mean,
                        )
                    )
                    color = next(colors)
                    color_darker = tuple(map(lambda c: c * 0.5, color))

                    # Plot fit distribution
                    sns.distplot(
                        points,
                        ax=axes[p_idx * 2],
                        label=name,
                        bins=args.bins,
                        kde=False,
                        norm_hist=True,
                        color=color
                    )
                    # Draw a darker solid line to represent the signal
                    axes[p_idx * 2].axvline(signal_coeffs[name][c_name], ymax=0.25, color=color_darker)
                    # Draw a darker dotted line to represent the mean
                    axes[p_idx * 2].axvline(mean, ymax=0.25, color=color_darker, linestyle=':')

                    # Plot fit pulls
                    sns.distplot(
                        pull,
                        ax=axes[p_idx * 2 + 1],
                        label=name,
                        bins=args.bins,
                        kde=False,
                        norm_hist=True,
                        color=color
                    )
                    # Draw a darker solid line to represent the pull mean
                    axes[p_idx * 2 + 1].axvline(np.mean(pull), ymax=0.25, color=color_darker)

                    drawn_something = True

            axes[p_idx * 2].set_xlim(-args.plot_scale, args.plot_scale)
            axes[p_idx * 2].set_ylabel('density')
            if drawn_something:
                axes[p_idx * 2].legend()
                axes[p_idx * 2 + 1].legend()

        plt.show()
