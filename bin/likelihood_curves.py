#!/usr/bin/env python
"""
Plot each coefficient vs. negative log likelihood whilst keeping other coefficients fixed at signal values.

The blue curve is the likelihood curve. The red line denotes the true signal value.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()


def try_nll(pos_, val):
    try_coeffs[pos_] = tf.constant(val)
    return bmf.signal.nll(try_coeffs, signal_events)


with bmf.Script() as script:
    signal_coeffs = bmf.coeffs.signal()
    signal_events = bmf.signal.generate(signal_coeffs)
    fit_coeffs = bmf.coeffs.fit()

    for a_idx in range(0, bmf.coeffs.amplitude_count):
        if not bmf.coeffs.is_trainable(fit_coeffs[a_idx*bmf.coeffs.param_count]):
            continue

        fig, axes = plt.subplots(bmf.coeffs.param_count)
        fig.suptitle(bmf.coeffs.amplitude_latex_names[a_idx])

        for p_idx in range(0, bmf.coeffs.param_count):
            c_idx = a_idx * bmf.coeffs.param_count + p_idx
            script.stdout('Processing {} ({})'.format(bmf.coeffs.names[c_idx], c_idx))
            try_coeffs = bmf.coeffs.signal()

            c_range = np.linspace(-12.0, 12.0, 100, dtype=np.float32)

            with tf.device('/device:GPU:0'):
                axes[p_idx].plot(c_range, list(map(lambda c_val: try_nll(c_idx, c_val).numpy() / 1e5, c_range)))

            axes[p_idx].set_ylabel(bmf.coeffs.param_latex_names[p_idx] + r' $(\times 10^5)$')
            axes[p_idx].axvline(signal_coeffs[c_idx].numpy(), ymax=0.5, color='r')

        plt.show()
