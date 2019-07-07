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

signal_events = bmf.signal.generate(100_000, bmf.coeffs.signal)


def try_nll1(pos_, val):
    try_coeffs[pos_] = tf.constant(val)
    return bmf.signal.nll(signal_events, try_coeffs)


for i in range(0, 8):
    fig, axes = plt.subplots(3)
    a_name = [
        r'Re($a_{\parallel}^L$)',
        r'Im($a_{\parallel}^L$)',
        r'Re($a_{\parallel}^R$)',
        r'Im($a_{\parallel}^R$)',
        r'Re($a_{\bot}^L$)',
        r'Im($a_{\bot}^L$)',
        r'Re($a_{\bot}^R$)',
        r'Re($a_{0}^L$)',
    ][i]
    fig.suptitle(a_name)

    for j in range(0, 3):
        pos = (i if i < 7 else 8) * 3 + j
        print('Processing {}'.format(bmf.coeffs.fit[pos].name[4:].split(':')[0]))
        try_coeffs = bmf.coeffs.signal.copy()

        c_range = np.linspace(-12.0, 12.0, 100, dtype=np.float32)

        axes[j].plot(c_range, list(map(lambda c_val: try_nll1(pos, c_val).numpy() / 1e5, c_range)))
        axes[j].set_ylabel([r'$\alpha$', r'$\beta$', r'$\gamma$'][j] + r' $(\times 10^5)$')

        axes[j].axvline(bmf.coeffs.signal[pos].numpy(), ymax=0.5, color='r')

    plt.show()

exit(0)


# C1 = 0
# C1_MIN = -3.45
# C1_MAX = -3.40
# C2 = 1
# C2_MIN = -0.15
# C2_MAX = -0.10
# CONTOUR_POINTS = 40
# SIGNAL_COUNT = 100_000
#
# fig, ax = plt.subplots()
#
# def try_nll2(_c1, _c2):
#     fit_coeffs[C1] = tf.constant(_c1)
#     fit_coeffs[C2] = tf.constant(_c2)
#     return negative_log_likelihood(signal_events, fit_coeffs)
#
#
# fit_coeffs = signal_coeffs
# c1 = np.linspace(C1_MIN, C1_MAX, CONTOUR_POINTS, dtype=np.float32)
# c2 = np.linspace(C2_MIN, C2_MAX, CONTOUR_POINTS, dtype=np.float32)
# nll = list(map(lambda _c2: list(map(lambda _c1: try_nll2(_c1, _c2).numpy(), c1)), c2))
#
# CS = ax.contour(c1, c2, nll)
# ax.clabel(CS, inline=1, fontsize=10)
# ax.set_title('Likelihood surface')
# plt.show()
