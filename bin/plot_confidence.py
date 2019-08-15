#!/usr/bin/env python
"""
Plot confidence, mean and signal values for an ensemble run
"""

import argparse
import csv
import matplotlib.pyplot as plt
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot confidence for CSV file.',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=columns, width=columns),
)
parser.add_argument(
    dest='file',
    type=str,
    metavar='FILENAME',
    help='Filename to plot'
)

args = parser.parse_args()


def param(row_, a_idx_, p_idx_):
    return tf.constant(
        float(
            row_[bmf.coeffs.names[(a_idx_ * bmf.coeffs.param_count) + p_idx_]]
        )
    )


def plots(row_):
    data_ = {}
    for a_idx_, amplitude_latex_name_ in enumerate(bmf.coeffs.amplitude_latex_names):
        alpha = param(row_, a_idx_, 0)
        beta = param(row_, a_idx_, 1)
        gamma = param(row_, a_idx_, 2)

        if alpha.numpy() == 0.0 and beta.numpy() == 0.0 and gamma.numpy() == 0.0:
            continue

        data_[amplitude_latex_name_] = alpha + (beta * q2_range) + (gamma / q2_range)
    return data_


with bmf.Script(device=None) as script:
    q2_range = tf.linspace(bmf.signal.q2_min, bmf.signal.q2_max, 200)

    # Load data
    with open(args.file, newline='') as csv_file:
        reader = csv.DictReader(csv_file)

        signal = next(reader)
        signal_plots = plots(signal)

        data_plots = {}
        for row in reader:
            p = plots(row)
            for amplitude_latex_name, plot in plots(row).items():
                if amplitude_latex_name not in data_plots:
                    data_plots[amplitude_latex_name] = []
                data_plots[amplitude_latex_name].append(plot)

    # Plot each amplitude
    for amplitude_latex_name in signal_plots:
        plt.title(amplitude_latex_name)

        tensor_data = tf.stack(data_plots[amplitude_latex_name], axis=0)

        mean = tf.reduce_mean(tensor_data, axis=0)
        plt.plot(q2_range.numpy(), mean.numpy(), color='magenta', label='mean')

        min_68 = []
        max_68 = []
        min_95 = []
        max_95 = []
        amplitude_max = 0.0
        for i, q2 in enumerate(q2_range):
            # For this q^2 value, get list of values above the mean
            above_mean = sorted(
                tf.squeeze(
                    tf.gather(
                        tensor_data[:, i],
                        tf.where(tf.math.greater(tensor_data[:, i], mean[i]))
                    )
                ).numpy()
            )
            max_68.append(above_mean[int((len(above_mean) - 1) * 0.68)])
            max_95.append(above_mean[int((len(above_mean) - 1) * 0.95)])

            # For this q^2 value, get list of values below the mean
            below_mean = sorted(
                tf.squeeze(
                    tf.gather(
                        tensor_data[:, i],
                        tf.where(tf.math.less(tensor_data[:, i], mean[i]))
                    )
                ).numpy(),
                reverse=True
            )
            min_68.append(below_mean[int((len(below_mean) - 1) * 0.68)])
            min_95.append(below_mean[int((len(below_mean) - 1) * 0.95)])

            # Keep track of the max so we can pick a sensible y-axis
            this_max = max(abs(min_68[-1]), abs(max_95[-1]))
            if this_max > amplitude_max:
                amplitude_max = this_max

        plt.fill_between(q2_range.numpy(), min_95, max_95, label='95%', color='yellow')
        plt.fill_between(q2_range.numpy(), min_68, max_68, label='68%', color='lightgreen')

        plt.plot(
            q2_range.numpy(),
            signal_plots[amplitude_latex_name].numpy(),
            color='black', label='signal', linestyle=':'
        )

        if amplitude_max > 0.0:
            plt.ylim(-amplitude_max * 1.1, amplitude_max * 1.1)

        plt.xlabel(r'$q^2 / (GeV^2/c^4)$')
        plt.legend()
        plt.show()
