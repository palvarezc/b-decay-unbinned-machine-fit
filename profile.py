import timeit

from coefficients import *
from signal_distribution import generate_events, negative_log_likelihood

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

signal_events = generate_events(100_000, signal_coeffs)

optimizer = tf.optimizers.Adam(learning_rate=0.01)

for t in [10, 100, 1000]:
    tf.print(
        "negative_log_likelihood x {}: ".format(t),
        timeit.timeit(
            lambda: negative_log_likelihood(signal_events, fit_coeffs),
            number=t
        )
    )

for t in [10, 100, 1000]:
    tf.print(
        "minimize x {}: ".format(t),
        timeit.timeit(
            lambda: optimizer.minimize(
                lambda: negative_log_likelihood(signal_events, fit_coeffs),
                var_list=trainable_coeffs,
            ),
            number=t
        )
    )