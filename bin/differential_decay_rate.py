#!/usr/bin/env python
"""
Plot the fraction of S-wave contribution for signal coefficients over the q^2 range
"""
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

with bmf.Script() as script:
    coeffs = bmf.coeffs.signal(bmf.coeffs.NP)
    q2 = tf.linspace(bmf.signal.q2_min, bmf.signal.q2_max, 100)
    amplitudes = bmf.signal.coeffs_to_amplitudes(coeffs, q2)

    decay_rate_angle_integrated = bmf.signal.decay_rate_angle_integrated(coeffs, q2)
    decay_rate_angle_integrated_p_wave = bmf.signal.decay_rate_angle_integrated_p_wave(amplitudes, q2)
    decay_rate_angle_integrated_s_wave = bmf.signal.decay_rate_angle_integrated_s_wave(amplitudes)

    modulus_frac_s = bmf.signal.modulus_frac_s(coeffs, q2)

    plt.plot(q2.numpy(), decay_rate_angle_integrated.numpy(), label='Combined')
    plt.plot(q2.numpy(), decay_rate_angle_integrated_p_wave.numpy(), label='P-wave')
    plt.plot(q2.numpy(), decay_rate_angle_integrated_s_wave.numpy(), label='S-wave')
    plt.xlabel(r'$q^2$ / GeV')
    plt.legend()
    plt.show()
