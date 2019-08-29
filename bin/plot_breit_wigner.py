#!/usr/bin/env python
"""
Print integrated BW distributions for K*0(892), K*0(700) and a mix of K*0(700)/K*0(892) between
+/- 100 MeV of K892 mass, and then plot distributions
"""
import argparse
import matplotlib
import shutil
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Plot BW distributions.',
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

    bmf.stdout(
        'Integrated values between +/- 100 MeV of K892 mass: K892: {} K700: {} Mix: {}'.format(
            bmf.breit_wigner.k892_distribution_integrated(),
            bmf.breit_wigner.k700_distribution_integrated(),
            bmf.breit_wigner.k700_k892_distribution_integrated()
        )
    )

    masses = tf.linspace(bmf.breit_wigner.mass_k_plus + bmf.breit_wigner.mass_pi_minus + 0.01, 2.0, 150)

    k700 = bmf.breit_wigner.k700_distribution(masses)
    k892 = bmf.breit_wigner.k892_distribution(masses)
    mix = tf.math.abs(bmf.breit_wigner.k700_k892_distribution(masses))

    plt.figure()
    sns.set(style='ticks')

    plt.plot(masses.numpy() * 1000, k892.numpy(), label=r'$K^*_0(892)$')
    plt.plot(masses.numpy() * 1000, k700.numpy(), label=r'$K^*_0(700)$')
    plt.plot(masses.numpy() * 1000, mix.numpy(), label=r'$K^*_0(892)/K^*_0(700)$')
    plt.xlabel('Mass / MeV')
    plt.ylabel('Absolute value')
    plt.margins(x=0)
    plt.legend()
    if args.write_svg is not None:
        filepath = args.write_svg
        bmf.stdout('Writing {}'.format(filepath))
        plt.savefig(filepath, format="svg")
    else:
        plt.show()
