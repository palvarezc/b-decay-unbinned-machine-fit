#!/usr/bin/env python
"""Fit amplitude coefficients to signal events"""

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()
# tf.debugging.set_log_device_placement(True)

#######################


def plot_signal(events):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Signal distributions')
    titles = [
        r'$q^2$',
        r'$\cos{\theta_k}$',
        r'$\cos{\theta_l}$',
        r'$\phi$'
    ]

    for ax, feature, title in zip(axes.flatten(), events.numpy().transpose(), titles):
        sns.distplot(feature, ax=ax, bins=20)
        ax.set(title=title)

    plt.show()


signal_events = bmf.signal.generate(bmf.coeffs.signal)
plot_signal(signal_events)

######################


def print_step(step):
    tf.print(
        "Step:", step,
        "nll:", nll,
        "grad_max:", grad_max,
        "grad_mean:", grad_mean,
        "grad_total:", grad_total,
        output_stream=sys.stdout
    )
    tf.print("fit:   ", bmf.coeffs.to_str(bmf.coeffs.fit), output_stream=sys.stdout)
    tf.print("signal:", bmf.coeffs.to_str(bmf.coeffs.signal), output_stream=sys.stdout)


# for i in range(18, 36):
#     bmf.coeffs.fit[i] = bmf.coeffs.signal[i]

grad_max = grad_mean = grad_total = None
nll = bmf.signal.nll(bmf.coeffs.signal, signal_events)
print_step("initial")
optimizer = tf.optimizers.Adam(learning_rate=0.20)

for i in range(10000):
    with tf.device('/device:GPU:0'):
        with tf.GradientTape() as tape:
            nll = bmf.signal.nll(bmf.coeffs.fit, signal_events)
        grads = tape.gradient(nll, bmf.coeffs.trainables())
        grad_max = tf.reduce_max(grads)
        grad_mean = tf.reduce_mean(grads)
        grad_total = tf.reduce_sum(grads)
        optimizer.apply_gradients(zip(grads, bmf.coeffs.trainables()))
    if i % 20 == 0:
        print_step(i)

print_step("final")
