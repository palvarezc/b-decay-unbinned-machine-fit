#!/usr/bin/env python
"""Plot generated signal for each independent variable"""

import argparse
import matplotlib
import numpy as np
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot generated signal for each independent variable.',
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
    default=100000,
    help='number of signal events to generated per plot (default: 100,000)'
)
parser.add_argument(
    '-S',
    '--signal-model',
    dest='signal_model',
    choices=bmf.coeffs.signal_models,
    default=bmf.coeffs.SM,
    help='signal model (default: {})'.format(bmf.coeffs.SM)
)
parser.add_argument(
    '-w',
    '--write-svg',
    dest='write_svg',
    metavar='SVG_PATH',
    help='write plots as SVGs using this filepath. this string must contain \'%%name%%\''
)
args = parser.parse_args()
if args.write_svg and '%name%' not in args.write_svg:
    parser.error('-w/--write-svg must contain \'%name%\'')

with bmf.Script(device=args.device) as script:
    if args.write_svg is not None:
        matplotlib.use('SVG')

    # Import these after we optionally set SVG backend - otherwise matplotlib may bail on a missing TK backend when
    #  running from the CLI
    import matplotlib.pylab as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    import seaborn as sns

    signal_coeffs = bmf.coeffs.signal(args.signal_model)
    signal_events = bmf.signal.generate(signal_coeffs, events_total=args.signal_count)

    names = ['q2', 'cos_theta_k', 'cos_theta_l', 'phi']
    latex_names = [
        r'$q^2 / (GeV^2 / c^4)$',
        r'$\cos{\theta_k}$',
        r'$\cos{\theta_l}$',
        r'$\phi$'
    ]
    for events, name, latex_name in zip(signal_events.numpy().transpose(), names, latex_names):
        plt.figure()
        # Set style as well as font to Computer Modern Roman to match LaTeX output
        sns.set(style='ticks', font='cmr10', rc={'mathtext.fontset': 'cm', 'axes.unicode_minus': False})

        sns.kdeplot(events, shade=True, cut=0)

        if name == 'phi':
            # Show phi x-axis ticks in units of pi/2
            plt.gca().xaxis.set_major_formatter(FuncFormatter(
                lambda val, pos: {
                    0: r'$-\pi$',
                    1: r'$\dfrac{-\pi}{2}$',
                    2: r'0',
                    3: r'$\dfrac{\pi}{2}$',
                    4: r'$\pi$',
                    5: r'_',
                    6: r'_',
                }[pos]
            ))
            plt.gca().xaxis.set_major_locator(MultipleLocator(base=np.pi / 2))

        plt.xlabel(latex_name)
        plt.margins(x=0)
        plt.ylabel('Event Density')
        if args.write_svg is not None:
            filepath = args.write_svg.replace('%name%', name)
            bmf.stdout('Writing {}'.format(filepath))
            plt.savefig(filepath, format="svg")
        else:
            plt.show()
