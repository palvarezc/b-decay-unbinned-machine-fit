#!/usr/bin/env python
"""
Print integrated BW distributions for K*0(892), K*0(700) and a mix of K*0(700)/K*0(892) between
+/- 100 MeV of K892 mass, and then plot distributions
"""
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

import b_meson_fit.breit_wigner as bmfbw
import b_meson_fit.script as bmfr

tf.enable_v2_behavior()

with bmfr.Script() as script:
    bmfr.stdout(
        'Integrated values between +/- 100 MeV of K892 mass: K892: {} K700: {} Mix: {}'.format(
            bmfbw.k892_distribution_integrated(),
            bmfbw.k700_distribution_integrated(),
            bmfbw.k700_k892_distribution_integrated()
        )
    )

    masses = tf.linspace(bmfbw.mass_k_plus + bmfbw.mass_pi_minus + 0.01, 2.0, 150)

    k700 = bmfbw.k700_distribution(masses)
    k892 = bmfbw.k892_distribution(masses)
    mix = tf.math.abs(bmfbw.k700_k892_distribution(masses))

    plt.plot(masses.numpy() * 1000, k700.numpy(), label=r'$\kappa^*_0(700)$')
    plt.plot(masses.numpy() * 1000, k892.numpy(), label=r'$K^*_0(892)$')
    plt.plot(masses.numpy() * 1000, mix.numpy(), label=r'mix')
    plt.xlabel('Mass / MeV')
    plt.legend()
    plt.show()
