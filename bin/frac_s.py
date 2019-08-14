#!/usr/bin/env python
"""
Plot the fraction of S-wave contribution for signal coefficients over the q^2 range
"""
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

with bmf.Script() as script:
    coeffs = bmf.coeffs.signal(bmf.coeffs.SM)

    q2 = tf.linspace(bmf.signal.q2_min, bmf.signal.q2_max, 100)
    decay_rate_frac_s = bmf.signal.decay_rate_frac_s(coeffs, q2)
    modulus_frac_s = bmf.signal.modulus_frac_s(coeffs, q2)

    plt.plot(q2.numpy(), decay_rate_frac_s.numpy(), label='decay_rate')
    plt.plot(q2.numpy(), modulus_frac_s.numpy(), label='modulus')
    plt.xlabel(r'$q^2$ / GeV')
    plt.ylabel(r'$F_s$')
    plt.legend()
    plt.show()
