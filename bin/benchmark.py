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
    signal_coeffs = bmf.coeffs.signal(bmf.coeffs.SM)
    signal_events = bmf.signal.generate(signal_coeffs)
    fit_coeffs = bmf.coeffs.fit(signal_coeffs)
    optimizer = bmf.Optimizer(fit_coeffs, signal_events)

    for n, f in functions.items():
        for t in times:
            time_taken = timeit.timeit(f, number=t)
            bmf.stdout("{}() x {}: ".format(n, t), time_taken)
