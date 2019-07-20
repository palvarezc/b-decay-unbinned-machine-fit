#!/usr/bin/env python
"""
The script generates profile information that can be viewed in Tensorboard.

See README.
"""
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

with bmf.Script() as script:
    if not bmf.user_is_root():
        bmf.stderr('This script needs root permissions. You can run it from the project folder with:')
        bmf.stderr(
            'sudo -E --preserve-env=PYTHONPATH ./bin/profile.py')
        exit(1)

    signal_coeffs = bmf.coeffs.signal()
    signal_events = bmf.signal.generate(signal_coeffs)
    log = bmf.Log(script.name)

    optimizer = bmf.Optimizer(
        bmf.coeffs.fit(),
        signal_events,
    )

    for i in range(1000):
        tf.summary.trace_on(graph=True, profiler=True)
        optimizer.minimize()
        tf.summary.trace_export(name='trace_%d' % optimizer.step, step=optimizer.step, profiler_outdir=log.dir())
        tf.summary.flush()
