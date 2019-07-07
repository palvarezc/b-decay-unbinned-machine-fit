#!/usr/bin/env python
"""
Profile time taken to run negative_log_likelihood() and minimize().

Used to check for performance regressions.
"""

import tensorflow.compat.v2 as tf
import timeit

import b_meson_fit as bmf

tf.enable_v2_behavior()

signal_events = bmf.signal.generate(100_000, bmf.coeffs.signal)

optimizer = tf.optimizers.Adam(learning_rate=0.01)

for t in [10, 100, 1000]:
    tf.print(
        "nll() x {}: ".format(t),
        timeit.timeit(
            lambda: bmf.signal.nll(signal_events, bmf.coeffs.fit),
            number=t
        )
    )

for t in [10, 100, 1000]:
    tf.print(
        "minimize() x {}: ".format(t),
        timeit.timeit(
            lambda: optimizer.minimize(
                lambda: bmf.signal.nll(signal_events, bmf.coeffs.fit),
                var_list=bmf.coeffs.trainable,
            ),
            number=t
        )
    )