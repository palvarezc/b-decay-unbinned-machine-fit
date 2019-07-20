#!/usr/bin/env python
"""
Plot the fraction of S-wave contribution for signal coefficients over the q^2 range
"""
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

with bmf.Script() as script:
    q2 = tf.linspace(bmf.signal.q2_min, bmf.signal.q2_max, 100)
    frac_s = bmf.signal.decay_rate_frac_s(bmf.coeffs.signal(), q2)

    plt.plot(q2.numpy(), frac_s.numpy())
    plt.xlabel(r'$q^2$')
    plt.ylabel(r'$F_s$')
    plt.show()
