#!/usr/bin/env python
"""Plot Q test statistics"""
import argparse
import csv
import math
import matplotlib
import numpy as np
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()


def read_q_stats(csv_path):
    """Return list of Q stats from file"""
    q_list = []
    with open(csv_path, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            q_list.append(float(row['q']))

    return q_list


def gaussian(x_list_, data):
    data_max = max(data)
    x = np.sum(x_list_ * data) / np.sum(data)
    width = np.sqrt(np.abs(np.sum((x_list_ - x) ** 2 * data) / np.sum(data)))
    return data_max * np.exp(-(x_list_ - x) ** 2 / (2 * width ** 2))


columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot Q test statistics.',
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
    help='write plot as SVG using this filepath'
)
parser.add_argument(
    dest='sm_filepath',
    metavar='SM_FILEPATH',
    help='Path to SM CSV file'
)
parser.add_argument(
    dest='np_filepath',
    metavar='NP_FILEPATH',
    help='Path to NP CSV file'
)

args = parser.parse_args()

with bmf.Script(device=args.device) as script:
    if args.write_svg is not None:
        matplotlib.use('SVG')

    # Import these after we optionally set SVG backend - otherwise matplotlib may bail on a missing TK backend when
    #  running from the CLI
    import matplotlib.pylab as plt
    import seaborn as sns

    # Load data
    sm_data = read_q_stats(args.sm_filepath)
    np_data = read_q_stats(args.np_filepath)

    # Max _/+ x-axis scale (Rounded up to nearest 25)
    combined_data = sm_data + np_data
    max_point = max(max(combined_data), -min(combined_data))
    x_max = 25 * math.ceil(max_point / 25)

    # Histogram bins to use
    bins = np.linspace(-x_max, x_max, args.bins)

    # Bin midpoints for x-axis
    x_list = (bins[1:] + bins[:-1]) / 2

    sm_hist = np.histogram(sm_data, bins=bins, density=True)
    np_hist = np.histogram(np_data, bins=bins, density=True)
    np_median = np.median(np_data)
    sm_gaussian = gaussian(x_list, sm_hist[0])

    # Calculate sigma confidence level
    sm_mean = np.mean(sm_data)
    sm_stddev = np.std(sm_data)
    sigma_level = (sm_mean - np_median) / sm_stddev
    bmf.stdout('mean: {} stddev: {} sigma level: {}'.format(sm_mean, sm_stddev, sigma_level))

    plt.figure()
    sns.set(style='ticks')

    # Blue open circles for SM data. Don't plot 0 values
    plt.scatter(x_list, [np.nan if x == 0 else x for x in sm_hist[0]], facecolors='none', edgecolors='b')

    # Red closed circles for NP data. Don't plot 0 values
    plt.scatter(x_list, [np.nan if x == 0 else x for x in np_hist[0]], color='r')

    # Blue solid line for SM Gaussian
    plt.plot(x_list, sm_gaussian, color='b')

    # Red dashed line for NP median
    plt.gca().axvline(np_median, color='r', linestyle=':')

    # Calculate the y-min by finding the y-axis order of magnitude just before the NP median, rounding down,
    #  and dropping 1 more order of magnitude
    x_idx_just_before_np_median = len(list(filter(lambda x: x < np_median, x_list))) - 1
    gaussian_val_just_before_np_median = sm_gaussian[x_idx_just_before_np_median]
    val_mag_just_before_np_median = math.floor(math.log(gaussian_val_just_before_np_median, 10))
    y_min = float('1e{}'.format(val_mag_just_before_np_median - 1))

    # Calculate the y-max by finding the highest y-axis and rounding up to the next order of magnitude
    max_fraction = max(sm_hist[0] + np_hist[0])
    max_fraction_mag = math.ceil(math.log(max_fraction, 10))
    y_max = float('1e{}'.format(max_fraction_mag))

    bmf.stdout('Setting x scale from {} to {}'.format(-x_max, x_max))
    plt.xlim(-x_max, x_max)
    bmf.stdout('Setting y scale from {} to {}'.format(y_min, y_max))
    plt.ylim(y_min, y_max)
    plt.xlabel('Q')
    plt.ylabel('Fraction / bin')
    plt.yscale('log')

    if args.write_svg is not None:
        filepath = args.write_svg
        bmf.stdout('Writing {}'.format(filepath))
        plt.savefig(filepath, format="svg")
    else:
        plt.show()


