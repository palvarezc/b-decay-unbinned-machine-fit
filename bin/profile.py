#!/usr/bin/env python
"""
Profile time taken to run key functions.

Used to check for performance regressions.
"""

import sys
import tensorflow.compat.v2 as tf
import timeit

import b_meson_fit as bmf

tf.enable_v2_behavior()

signal_events = bmf.signal.generate(bmf.coeffs.signal)

optimizer = tf.optimizers.Adam(learning_rate=0.01)

times = [10, 100, 1000]
functions = {
    "nll": lambda: bmf.signal.nll(signal_events, bmf.coeffs.fit),
    "minimise": lambda: optimizer.minimize(
        lambda: bmf.signal.nll(signal_events, bmf.coeffs.fit),
        var_list=bmf.coeffs.trainables(),
    )
}

for n, f in functions.items():
    for t in times:
        with tf.device('/device:GPU:0'):
            time_taken = timeit.timeit(f, number=t)
        tf.print("{}() x {}: ".format(n, t), time_taken, output_stream=sys.stdout)
