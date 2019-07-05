import matplotlib.pyplot as plt
import seaborn as sns
import sys

from coefficients import *
from signal_distribution import generate_events, negative_log_likelihood

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

# tf.debugging.set_log_device_placement(True)

# TODO: Optimise integral
# TODO: Check maths terms

# TODO: Switch to accept-reject/monte-carlo. Increase sample size
# TODO: Fix fitting/Check one or two parameter fits+surfaces

# TODO: Fix basis fitting/Check

# TODO: Do ensembles & plot distributions. Change to proper sample size
# TODO: Optimise hyperparameters, choose optimiser, more layers? Keras?
# TODO: Convert to tf distribution?/Unit tests/comments/doc comments

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


signal_events = generate_events(100_000, signal_coeffs)
plot_signal(signal_events)

######################


def print_step(step):
    tf.print("Step:", step, "nll:", negative_log_likelihood(signal_events, fit_coeffs), output_stream=sys.stdout)
    tf.print("fit:   ", coeffs_to_string(fit_coeffs), output_stream=sys.stdout)
    tf.print("signal:", coeffs_to_string(signal_coeffs), output_stream=sys.stdout)


print_step("initial")
optimizer = tf.optimizers.Nadam(learning_rate=0.01)

for i in range(10000):
    optimizer.minimize(lambda: negative_log_likelihood(signal_events, fit_coeffs), var_list=trainable_coeffs)
    if i % 20 == 0:
        print_step(i)

print_step("final")
