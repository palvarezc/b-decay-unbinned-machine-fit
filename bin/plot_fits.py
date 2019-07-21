#!/usr/bin/env python
"""
Plot histograms of CSV result files.
"""

import argparse
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

sns.set_palette("Set2")


def name_and_filename(arg):
    try:
        _name, _filename = arg.split(":")
    except ValueError:
        raise argparse.ArgumentError(None, "Plot list must be specified as NAME:FILENAME")
    return _name, _filename


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
    metavar='NAME:FILENAME',
    help='Name and filename pairs to plot'
)

args = parser.parse_args()
print(args)

with bmf.Script(device=None) as script:
    data_points = {}
    for plot in args.plot_list[0]:
        p_name, filename = plot
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(
                csvfile,
                fieldnames=(['id', 'normalized_nll'] + bmf.coeffs.names + ['time_taken'])
            )
            next(reader, None)  # Skip headers
            for row in reader:
                for c_name in bmf.coeffs.names:
                    if c_name not in data_points:
                        data_points[c_name] = {}
                    if p_name not in data_points[c_name]:
                        data_points[c_name][p_name] = []
                    data_points[c_name][p_name].append(float(row[c_name]))

    signal_coeffs = bmf.coeffs.signal()

    # For each amplitude
    for a_idx in range(0, bmf.coeffs.amplitude_count):
        fig, axes = plt.subplots(bmf.coeffs.param_count)
        fig.suptitle(bmf.coeffs.amplitude_latex_names[a_idx])

        # For each param in this amplitude
        for p_idx in range(0, bmf.coeffs.param_count):
            c_idx = a_idx * bmf.coeffs.param_count + p_idx

            bmf.stdout('Processing {} ({})'.format(bmf.coeffs.names[c_idx], c_idx))

            drawn_something = False
            for name, points in data_points[bmf.coeffs.names[c_idx]].items():
                if not all(elem == 0.0 for elem in points):
                    sns.distplot(points, ax=axes[p_idx], label=name, bins=args.bins, kde=False)
                    drawn_something = True
            axes[p_idx].set_xlim(-args.plot_scale, args.plot_scale)
            axes[p_idx].set_ylabel('count')
            if drawn_something:
                axes[p_idx].legend()
                # Draw a red line to represent the signal
                axes[p_idx].axvline(signal_coeffs[c_idx].numpy(), ymax=0.25, color='r')

        plt.show()
