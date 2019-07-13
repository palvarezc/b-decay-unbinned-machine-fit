#!/usr/bin/env python
"""Fit amplitude coefficients to signal events"""

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

log = False

with bmf.Script() as script:
    signal_coeffs = bmf.coeffs.signal()
    fit_coeffs = bmf.coeffs.fit()
    if log:
        log = bmf.Log(script.name)

    signal_events = bmf.signal.generate(signal_coeffs)

    # Plot our signal distributions for each independent variable
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Signal distributions')
    titles = [
        r'$q^2$',
        r'$\cos{\theta_k}$',
        r'$\cos{\theta_l}$',
        r'$\phi$'
    ]
    for ax, feature, title in zip(axes.flatten(), signal_events.numpy().transpose(), titles):
        sns.distplot(feature, ax=ax, bins=20)
        ax.set(title=title)
    plt.show()

    optimizer = bmf.Optimizer(fit_coeffs, signal_events)

    def print_step():
        bmf.stdout(
            "Step:", optimizer.step,
            "normalized_nll:", optimizer.normalized_nll,
            "grad_max:", optimizer.grad_max,
        )
        bmf.stdout("fit:   ", bmf.coeffs.to_str(fit_coeffs))
        bmf.stdout("signal:", bmf.coeffs.to_str(signal_coeffs))

    print_step()

    while True:
        optimizer.minimize()
        if log:
            log.coefficients('fit', optimizer, signal_coeffs)
        if optimizer.converged():
            break
        if optimizer.step.numpy() % 100 == 0:
            print_step()

    print_step()
