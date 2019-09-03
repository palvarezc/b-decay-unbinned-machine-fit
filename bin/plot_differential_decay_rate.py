#!/usr/bin/env python
"""
Plot the differential decay rate over the q^2 range
"""
import argparse
import matplotlib
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot differential decay rate.',
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
    help='write plot as SVG using this filepath'
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

    decay_rate_angle_integrated = bmf.signal.decay_rate_angle_integrated(coeffs, q2)
    decay_rate_angle_integrated_p_wave = bmf.signal.decay_rate_angle_integrated_p_wave(amplitudes, q2)
    decay_rate_angle_integrated_s_wave = bmf.signal.decay_rate_angle_integrated_s_wave(amplitudes)

    plt.figure()
    # Set style as well as font to Computer Modern Roman to match LaTeX output
    sns.set(style='ticks', font='cmr10', rc={'mathtext.fontset': 'cm', 'axes.unicode_minus': False})

    plt.plot(q2.numpy(), decay_rate_angle_integrated.numpy(), label='Combined')
    plt.plot(q2.numpy(), decay_rate_angle_integrated_p_wave.numpy(), label='P-wave')
    plt.plot(q2.numpy(), decay_rate_angle_integrated_s_wave.numpy(), label='S-wave')
    plt.xlabel(r'$q^2 / (GeV^2/c^4)$')
    plt.margins(x=0)
    plt.legend()
    if args.write_svg is not None:
        filepath = args.write_svg
        bmf.stdout('Writing {}'.format(filepath))
        plt.savefig(filepath, format='svg', bbox_inches='tight')
    else:
        plt.show()
