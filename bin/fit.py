#!/usr/bin/env python
"""Fit amplitude coefficients to signal events"""

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

with bmf.Script(log=True) as script:
    signal_coeffs = bmf.coeffs.signal()
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

    optimizer = bmf.Optimizer(
        script,
        bmf.coeffs.fit(),
        signal_events,
        'Adam',
        signal_coeffs=signal_coeffs,
        learning_rate=0.20
    )

    optimizer.print_step()

    while True:
        optimizer.minimize()
        if optimizer.converged():
            break
        if optimizer.step % 100 == 0:
            optimizer.print_step()

    optimizer.print_step()
