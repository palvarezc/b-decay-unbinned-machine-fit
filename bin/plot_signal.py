#!/usr/bin/env python
"""Plot generated signal for q^2 and angles"""

import argparse
import os
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

# Only do plots if running PyCharm
if 'PYCHARM_HOSTED' in os.environ:
    import matplotlib.pylab as plt
    import seaborn as sns

tf.enable_v2_behavior()

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot generated signal for q^2 and angles.',
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
    '-s',
    '--signal-count',
    dest='signal_count',
    type=int,
    default=2400,
    help='number of signal events to generated per fit (default: 2400)'
)
parser.add_argument(
    '-S',
    '--signal-model',
    dest='signal_model',
    choices=bmf.coeffs.signal_models,
    default=bmf.coeffs.SM,
    help='signal model (default: {})'.format(bmf.coeffs.SM)
)
args = parser.parse_args()

with bmf.Script(device=args.device) as script:
    signal_coeffs = bmf.coeffs.signal(args.signal_model)
    signal_events = bmf.signal.generate(signal_coeffs, events_total=args.signal_count)

    # If running if PyCharm, plot our signal distributions for each independent variable
    if 'PYCHARM_HOSTED' in os.environ:
        titles = [
            r'$q^2 / (GeV^2 / c^4)$',
            r'$\cos{\theta_k}$',
            r'$\cos{\theta_l}$',
            r'$\phi$'
        ]
        for feature, title in zip(signal_events.numpy().transpose(), titles):
            sns.distplot(feature, bins=20)
            plt.xlabel(title)
            plt.ylabel('Density')
            plt.show()
