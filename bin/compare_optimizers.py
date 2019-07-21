#!/usr/bin/env python
"""
Generate metrics for different optimizers and learning rates for comparison in Tensorboard
"""
import tensorflow.compat.v2 as tf
import tqdm

import b_meson_fit as bmf

tf.enable_v2_behavior()

# Test name, Optimizer name, Learning rate, Additional optimizer params, Gradient clip
combos = [
    ['Adam_0.1_noclip_nocutoff', 'Adam', 0.1, {}, None],
    ['AmsGrad_0.1_noclip_nocutoff', 'AMSGrad', 0.2, {}, None],
    ['AmsGrad_0.2_noclip_nocutoff', 'AMSGrad', 0.2, {}, None],
]
iterations = 2000
# Set all default fit coefficients to the same value to make comparison possible
bmf.coeffs.fit_default = 1.0

with bmf.Script() as script:
    signal_coeffs = bmf.coeffs.signal()
    signal_events = bmf.signal.generate(signal_coeffs)

    log = bmf.Log(script.name)

    # Draw a signal line on each coefficient plot so we can compare how well the optimizers do
    log.signal_line(bmf.coeffs.fit(), signal_coeffs, iterations)

    for combo in combos:
        test_name, name, learning_rate, params, clip = combo

        optimizer = bmf.Optimizer(
            bmf.coeffs.fit(),  # Generate new fit coefficients for each run
            signal_events,
            opt_name=name,
            learning_rate=learning_rate,
            opt_args=params,
            grad_clip=clip,
        )

        # Use tqdm's trange() to print a progress bar for each optimizer/learning rate combo
        with tqdm.trange(iterations, desc=test_name) as t:
            for i in t:
                optimizer.minimize()
                log.coefficients(test_name, optimizer, signal_coeffs)
