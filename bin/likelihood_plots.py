#!/usr/bin/env python
"""
Plot each coefficient vs. negative log likelihood whilst keeping other coefficients fixed at signal values.

If working properly each plot should show a minimum.
"""

import matplotlib.pyplot as plt
import numpy as np

from b_meson_fit.coefficients import fit_coeffs as real_fit_coeffs, signal_coeffs
from b_meson_fit.signal_distribution import generate_events, negative_log_likelihood

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

signal_events = generate_events(100_000, signal_coeffs)
fit_coeffs = signal_coeffs

##############################


def try_nll1(pos_, val):
    fit_coeffs[pos_] = tf.constant(val)
    return negative_log_likelihood(signal_events, fit_coeffs)


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
        print('Processing {}'.format(real_fit_coeffs[pos].name[4:].split(':')[0]))
        fit_coeffs = signal_coeffs

        # sig_val = signal_coeffs[pos].numpy()
        # min_ = sig_val - 5.0
        # max_ = sig_val + 5.0
        min_ = -100.0  # sig_val - 5.0
        max_ = 100.0  # sig_val + 5.0

        c_range = np.linspace(min_, max_, 200, dtype=np.float32)

        axes[j].plot(c_range, list(map(lambda c_val: try_nll1(pos, c_val).numpy() / 1e5, c_range)))
        axes[j].set_ylabel([r'$\alpha$', r'$\beta$', r'$\gamma$'][j] + r' $(\times 10^5)$')

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
