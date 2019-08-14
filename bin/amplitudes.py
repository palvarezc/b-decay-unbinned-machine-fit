#!/usr/bin/env python
"""
Plot amplitudes for NP signal coefficients
"""
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

with bmf.Script() as script:
    coeffs = bmf.coeffs.signal(bmf.coeffs.NP)
    q2 = tf.linspace(bmf.signal.q2_min, bmf.signal.q2_max, 100)
    amplitudes = bmf.signal.coeffs_to_amplitudes(coeffs, q2)

    for amplitude in amplitudes:
        real = tf.math.real(amplitude)
        imag = tf.math.imag(amplitude)

        real_latex_name = bmf.coeffs.amplitude_latex_names[(amplitudes.index(amplitude) * 2)]
        imag_latex_name = bmf.coeffs.amplitude_latex_names[(amplitudes.index(amplitude) * 2) + 1]

        plt.plot(q2.numpy(), real.numpy(), label=real_latex_name)
        plt.plot(q2.numpy(), imag.numpy(), label=imag_latex_name)
        plt.xlabel(r'$q^2$ / GeV')
        plt.legend()
        plt.show()
