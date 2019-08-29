#!/usr/bin/env python
"""
Plot amplitudes from signal coefficients
"""
import argparse
import matplotlib
import re
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot amplitudes from signal coefficients.',
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

with bmf.Script(device=args.device) as script:
    if args.write_svg is not None:
        matplotlib.use('SVG')

    # Import these after we optionally set SVG backend - otherwise matplotlib may bail on a missing TK backend when
    #  running from the CLI
    import matplotlib.pylab as plt
    import seaborn as sns

    coeffs = bmf.coeffs.signal(args.signal_model)
    q2 = tf.linspace(bmf.signal.q2_min, bmf.signal.q2_max, 100)
    amplitudes = bmf.signal.coeffs_to_amplitudes(coeffs, q2)

    for amplitude in amplitudes:
        real = tf.math.real(amplitude)
        imag = tf.math.imag(amplitude)

        name = re.sub('_re$', '', bmf.coeffs.amplitude_names[(amplitudes.index(amplitude) * 2)])
        real_latex_name = bmf.coeffs.amplitude_latex_names[(amplitudes.index(amplitude) * 2)]
        imag_latex_name = bmf.coeffs.amplitude_latex_names[(amplitudes.index(amplitude) * 2) + 1]

        plt.figure()
        sns.set(style='ticks')

        plt.plot(q2.numpy(), real.numpy(), label=real_latex_name)
        plt.plot(q2.numpy(), imag.numpy(), label=imag_latex_name)
        plt.xlabel(r'$q^2 / (GeV^2/c^4)$')
        plt.margins(x=0)
        plt.legend()

        if args.write_svg is not None:
            filepath = args.write_svg.replace('%name%', name)
            bmf.stdout('Writing {}'.format(filepath))
            plt.savefig(filepath, format="svg")
        else:
            plt.show()
