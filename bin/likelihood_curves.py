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

signal_events = bmf.signal.generate(bmf.coeffs.signal)


def try_nll(pos_, val):
    try_coeffs[pos_] = tf.constant(val)
    return bmf.signal.nll(signal_events, try_coeffs)


for i in range(0, 8):
    fig, axes = plt.subplots(3)
    a_name = bmf.coeffs.amplitude_latex_names[i]
    fig.suptitle(a_name)

    for j in range(0, 3):
        pos = (i if i < 7 else 8) * 3 + j
        print('Processing {}'.format(bmf.coeffs.fit[pos].name[4:].split(':')[0]))
        try_coeffs = bmf.coeffs.signal.copy()

        c_range = np.linspace(-12.0, 12.0, 100, dtype=np.float32)

        axes[j].plot(c_range, list(map(lambda c_val: try_nll(pos, c_val).numpy() / 1e5, c_range)))
        axes[j].set_ylabel(bmf.coeffs.param_latex_names[j] + r' $(\times 10^5)$')

        axes[j].axvline(bmf.coeffs.signal[pos].numpy(), ymax=0.5, color='r')

    plt.show()
