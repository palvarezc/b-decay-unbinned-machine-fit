#!/usr/bin/env python
"""
Benchmark time taken to run key functions.

Used to check for performance regressions.
"""
import tensorflow.compat.v2 as tf
import timeit

import b_meson_fit as bmf

tf.enable_v2_behavior()

times = [10, 100, 1000]
functions = {
    "nll": lambda: bmf.signal.nll(fit_coeffs, signal_events),
    "minimize": lambda: optimizer.minimize()
}

with bmf.Script() as script:
    signal_events = bmf.signal.generate(bmf.coeffs.signal())
    fit_coeffs = bmf.coeffs.fit()
    optimizer = bmf.Optimizer(script, fit_coeffs, signal_events, 'Adam', learning_rate=0.10)

    for n, f in functions.items():
        for t in times:
            time_taken = timeit.timeit(f, number=t)
            script.stdout("{}() x {}: ".format(n, t), time_taken)
