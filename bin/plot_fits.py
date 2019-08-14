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
import scipy as sp
import seaborn as sns
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()


def name_and_filename(arg):
    try:
        _name, _signal_model, _filename = arg.split(":")
    except ValueError:
        raise argparse.ArgumentError(None, "Plot list must be specified as NAME:SIGNAL_MODEL:FILENAME")
    if _signal_model not in bmf.coeffs.signal_models:
        raise argparse.ArgumentError(
            None,
            "Signal model {} must be one of {}".format(_signal_model, ','.join(bmf.coeffs.signal_models))
        )
    return _name, _signal_model, _filename


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
    type=name_and_filename,
    metavar='NAME:SIGNAL_MODEL:FILENAME',
    help='Name, signal model and filename to plot (e.g. NP_0.15:NP:NP_0.15.csv)'
)

args = parser.parse_args()
print(args)

with bmf.Script(device=None) as script:
    # Load inputs
    data_points = {}
    signal_coeffs = {}
    for plot in args.plot_list[0]:
        p_name, signal_model, filename = plot
        signal_coeffs[p_name] = bmf.coeffs.signal(signal_model)
        with open(filename, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
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
            colors = itertools.cycle(sns.color_palette("Set2"))

            drawn_something = False
            for name, points in data_points[bmf.coeffs.names[c_idx]].items():
                if not all(elem == 0.0 for elem in points):
                    mean = np.mean(points)
                    std_err = sp.stats.sem(points, axis=None)
                    pull = list(map(lambda p: (p - signal_coeffs[name][c_idx].numpy()) / std_err, points))
                    pull_mean = np.mean(points)

                    bmf.stdout(
                        '{}/{} signal: {} mean: {} std err: {} pull mean: {}'.format(
                            bmf.coeffs.names[c_idx],
                            name,
                            signal_coeffs[name][c_idx].numpy(),
                            mean,
                            std_err,
                            pull_mean,
                        )
                    )
                    color = next(colors)
                    color_darker = tuple(map(lambda c: c * 0.5, color))

                    # Fit distribution
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
                    axes[p_idx * 2].axvline(signal_coeffs[name][c_idx].numpy(), ymax=0.25, color=color_darker)
                    # Draw a darker dotted line to represent the mean
                    axes[p_idx * 2].axvline(mean, ymax=0.25, color=color_darker, linestyle=':')

                    # Fit pull
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
