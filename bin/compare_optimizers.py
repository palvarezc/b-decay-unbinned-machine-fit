#!/usr/bin/env python
"""
Generate metrics for different optimizers and learning rates for comparison in Tensorboard
"""
import tensorflow.compat.v2 as tf
from tensorflow.python.util import deprecation
from tqdm import trange

import b_meson_fit as bmf

# Force deprecation warnings off to stop them breaking our progress bars. The warnings are from TF internal code anyway.
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.enable_v2_behavior()

# Test name, Optimizer name, Optimizer params, Gradient clip, Gradient cutoff, Cutoff count, Cutoff value
combos = [
    ['Adam_0.1_noclip_nocutoff', 'Adam', {'learning_rate': 0.1}, None, False, None, None],
    ['Adam_0.2_noclip_nocutoff', 'Adam', {'learning_rate': 0.2}, None, False, None, None],
    ['Adam_0.2_clip_5.0_nocutoff', 'Adam', {'learning_rate': 0.2}, 5.0, False, None, None],
    ['Adam_0.2_clip_5.0_defaultcutoff', 'Adam', {'learning_rate': 0.2}, 5.0, True, None, None],
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
        test_name, name, params, clip, cutoff, cutoff_count, cutoff_value = combo

        optimizer = bmf.Optimizer(
            bmf.coeffs.fit(),  # Generate new fit coefficients for each run
            signal_events,
            opt_name=name,
            opt_args=params,
            grad_clip=clip,
            grad_cutoff=cutoff,
            grad_cutoff_count=cutoff_count,
            grad_cutoff_value=cutoff_value
        )

        # Use tqdm's trange() to print a progress bar for each optimizer/learning rate combo
        with trange(iterations, desc=test_name) as t:
            for i in t:
                optimizer.minimize()
                log.coefficients(test_name, optimizer, signal_coeffs)
