#!/usr/bin/env python
"""
Plot angular observables for signal coefficients
"""
import argparse
import matplotlib
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot angular observables for signal coefficients.',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=columns, width=columns),
)
parser.add_argument(
    '-d',
    '--device',
    dest='device',
    default=bmf.Script.device_default,
    help='use this device e.g. CPU:0, GPU:0, GPU:1 (default, {})'.format(bmf.Script.device_default),
)
parser.add_argument(
    '-S',
    '--signal-model',
    dest='signal_model',
    choices=bmf.coeffs.signal_models,
    default=bmf.coeffs.SM,
    help='signal model (default, {})'.format(bmf.coeffs.SM)
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

    # Mass terms
    four_mass2_over_q2 = bmf.signal.four_mass2_over_q2(q2)
    beta2 = bmf.signal.beta2_mu(four_mass2_over_q2)
    beta = tf.sqrt(beta2)

    # Observables
    observables = [
        ('j1s', r'$j_{1s}$', bmf.signal.j1s(amplitudes, beta2, four_mass2_over_q2)),
        ('j1c', r'$j_{1c}$', bmf.signal.j1c(amplitudes, four_mass2_over_q2)),
        ('j2s', r'$j_{2s}$', bmf.signal.j2s(amplitudes, beta2)),
        ('j2c', r'$j_{2c}$', bmf.signal.j2c(amplitudes, beta2)),
        ('j3', r'$j_3$', bmf.signal.j3(amplitudes, beta2)),
        ('j4', r'$j_4$', bmf.signal.j4(amplitudes, beta2)),
        ('j5', r'$j_5$', bmf.signal.j5(amplitudes, beta)),
        ('j6s', r'$j_{6s}$', bmf.signal.j6s(amplitudes, beta)),
        ('j7', r'$j_7$', bmf.signal.j7(amplitudes, beta)),
        ('j8', r'$j_8$', bmf.signal.j8(amplitudes, beta2)),
        ('j9', r'$j_9$', bmf.signal.j9(amplitudes, beta2)),
        ('jp1c', r'$j_{1c}^{\prime}$', bmf.signal.j1c_prime(amplitudes)),
        ('jpp1c', r'$j_{1c}^{\prime\prime}$', bmf.signal.j1c_dblprime(amplitudes)),
        ('jp4', r'$j_4^{\prime}$', bmf.signal.j4_prime(amplitudes)),
        ('jp5', r'$j_5^{\prime}$', bmf.signal.j5_prime(amplitudes)),
        ('jp7', r'$j_7^{\prime}$', bmf.signal.j7_prime(amplitudes)),
        ('jp8', r'$j_8^{\prime}$', bmf.signal.j8_prime(amplitudes)),
    ]

    for observable in observables:
        name, latex_name, values = observable
        plt.figure()
        sns.set(style='ticks')

        plt.plot(q2.numpy(), values.numpy())
        plt.margins(x=0)
        plt.title(latex_name)
        plt.xlabel(r'$q^2 / (GeV^2 / c^4)$')
        if args.write_svg is not None:
            filepath = args.write_svg.replace('%name%', name)
            bmf.stdout('Writing {}'.format(filepath))
            plt.savefig(filepath, format="svg")
        else:
            plt.show()
