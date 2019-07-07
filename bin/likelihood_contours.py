#!/usr/bin/env python
"""
Plot negative log likelihood surface for two coefficients whilst keeping others fixed at signal values.

The red lines mark the true signal values
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

cx_idx = 2
cy_idx = 5
grid_points = 50
signal_events = bmf.signal.generate(bmf.coeffs.signal)


def try_nll(c):
    fit_coeffs[cx_idx] = tf.constant(c[0])
    fit_coeffs[cy_idx] = tf.constant(c[1])
    return bmf.signal.nll(signal_events, fit_coeffs)


fit_coeffs = bmf.coeffs.signal.copy()
cx = np.linspace(-12.0, 12.0, grid_points, dtype=np.float32)
cy = np.linspace(-12.0, 12.0, grid_points, dtype=np.float32)
X, Y = tf.meshgrid(cx, cy)

points_grid = tf.stack([X, Y], axis=2)
points = tf.reshape(points_grid, [grid_points ** 2, 2])
likelihoods = tf.map_fn(try_nll, points)
likelihoods_grid = tf.reshape(likelihoods, [grid_points, grid_points])

fig, ax = plt.subplots()
CS = ax.contour(X, Y, likelihoods_grid / 1e5)
ax.set_title(r'Likelihood $(\times 10^5)$')
ax.clabel(CS, inline=1, fontsize=10)

ax.set_xlabel(bmf.coeffs.latex_name(cx_idx))
ax.set_ylabel(bmf.coeffs.latex_name(cy_idx))

ax.axvline(bmf.coeffs.signal[cx_idx].numpy(), color='r')
ax.axhline(bmf.coeffs.signal[cy_idx].numpy(), color='r')

plt.show()
