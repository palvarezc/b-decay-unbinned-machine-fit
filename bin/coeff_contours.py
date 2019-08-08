#!/usr/bin/env python
"""
Plot negative log likelihood surface for two coefficients whilst keeping others fixed at signal values.

The red lines mark the true signal values

Used to test that coefficients show minimums in correct places.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

cx_idx = 2
cy_idx = 5
grid_points = 50
grid_min = -12.0
grid_max = 12.0


# Use autograph for performance
@tf.function
def nll(coeffs_, signal_events_):
    return bmf.signal.normalized_nll(coeffs_, signal_events_)


def try_nll(c):
    # Override the two coeffs we want to vary
    fit_coeffs[cx_idx] = tf.constant(c[0])
    fit_coeffs[cy_idx] = tf.constant(c[1])
    return nll(fit_coeffs, signal_events)


with bmf.Script():
    signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
    signal_events = bmf.signal.generate(signal_coeffs)
    # Set fit coeffs to our constant signal ones
    fit_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)

    cx = np.linspace(grid_min, grid_max, grid_points, dtype=np.float32)
    cy = np.linspace(grid_min, grid_max, grid_points, dtype=np.float32)
    X, Y = tf.meshgrid(cx, cy)

    points_grid = tf.stack([X, Y], axis=2)
    # Turn our grid into (x, y) pairs
    points = tf.reshape(points_grid, [grid_points ** 2, 2])
    # Calculate likelihoods
    likelihoods = tf.map_fn(try_nll, points)
    # Convert likelihoods back into meshgrid shape
    likelihoods_grid = tf.reshape(likelihoods, [grid_points, grid_points])

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, likelihoods_grid)

    # Label contours
    ax.clabel(CS, inline=1, fontsize=10)

    # Label axes
    ax.set_xlabel(bmf.coeffs.latex_names[cx_idx])
    ax.set_ylabel(bmf.coeffs.latex_names[cy_idx])

    # Draw X/Y signal lines
    ax.axvline(signal_coeffs[cx_idx].numpy(), color='r')
    ax.axhline(signal_coeffs[cy_idx].numpy(), color='r')

    plt.show()
