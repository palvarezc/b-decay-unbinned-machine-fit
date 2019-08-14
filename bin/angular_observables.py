#!/usr/bin/env python
"""
Plot angular observables for NP signal coefficients
"""
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

with bmf.Script() as script:
    coeffs = bmf.coeffs.signal(bmf.coeffs.NP)
    q2 = tf.linspace(bmf.signal.q2_min, bmf.signal.q2_max, 100)
    amplitudes = bmf.signal.coeffs_to_amplitudes(coeffs, q2)

    # Mass terms
    four_mass2_over_q2 = bmf.signal.four_mass2_over_q2(q2)
    beta2 = bmf.signal.beta2_mu(four_mass2_over_q2)
    beta = tf.sqrt(beta2)

    # Observables
    observables = {
        r'$j_{1s}$': bmf.signal.j1s(amplitudes, beta2, four_mass2_over_q2),
        r'$j_{1c}$': bmf.signal.j1c(amplitudes, four_mass2_over_q2),
        r'$j_{2s}$': bmf.signal.j2s(amplitudes, beta2),
        r'$j_{2c}$': bmf.signal.j2c(amplitudes, beta2),
        r'$j_3$': bmf.signal.j3(amplitudes, beta2),
        r'$j_4$': bmf.signal.j4(amplitudes, beta2),
        r'$j_5$': bmf.signal.j5(amplitudes, beta),
        r'$j_{6s}$': bmf.signal.j6s(amplitudes, beta),
        r'$j_7$': bmf.signal.j7(amplitudes, beta),
        r'$j_8$': bmf.signal.j8(amplitudes, beta2),
        r'$j_9$': bmf.signal.j9(amplitudes, beta2),
        r'$j_{1c}^{\prime}$': bmf.signal.j1c_prime(amplitudes),
        r'$j_{1c}^{\prime\prime}$': bmf.signal.j1c_dblprime(amplitudes),
        r'$j_4^{\prime}$': bmf.signal.j4_prime(amplitudes),
        r'$j_5^{\prime}$': bmf.signal.j5_prime(amplitudes),
        r'$j_7^{\prime}$': bmf.signal.j7_prime(amplitudes),
        r'$j_8^{\prime}$': bmf.signal.j8_prime(amplitudes),
    }

    for latex_name, values in observables.items():
        plt.plot(q2.numpy(), values.numpy(), label=latex_name)
        plt.xlabel(r'$q^2$ / GeV')
        plt.legend()
        plt.show()
