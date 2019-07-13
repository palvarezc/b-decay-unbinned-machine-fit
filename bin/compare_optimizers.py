#!/usr/bin/env python
"""
Generate metrics for different optimizers and learning rates for comparison in Tensorboard

Once the script starts it will print out how to start Tensorboard, and the filter regex that can be
inputted in the left pane under 'Runs' to filter out just this run.
"""
import tensorflow.compat.v2 as tf
from tensorflow.python.util import deprecation
from tqdm import trange

import b_meson_fit as bmf

# Force deprecation warnings off to stop them breaking our progress bars. The warnings are from TF internal code anyway.
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.enable_v2_behavior()

# Test name, Optimizer name, Optimizer params, Gradient clip
combos = [
    ['Adam_0.1_noclip', 'Adam', {'learning_rate': 0.1}, None],
    ['Adam_0.2_noclip', 'Adam', {'learning_rate': 0.2}, None],
    ['Adam_0.2_clip_5.0', 'Adam', {'learning_rate': 0.2}, 5.0],
]
iterations = 2000

with bmf.Script() as script:
    signal_coeffs = bmf.coeffs.signal()
    signal_events = bmf.signal.generate(signal_coeffs)

    log = bmf.Log(script.name)

    # Draw a signal line on each coefficient plot so we can compare how well the optimizers do
    log.signal_line(bmf.coeffs.fit(), signal_coeffs, iterations)

    for combo in combos:
        test_name, name, params, clip = combo

        optimizer = bmf.Optimizer(
            bmf.coeffs.fit(),  # Generate new fit coefficients for each run
            signal_events,
            opt_name=name,
            opt_args=params,
            grad_clip=clip,
        )

        # Use tqdm's trange() to print a progress bar for each optimizer/learning rate combo
        with trange(iterations, desc=test_name) as t:
            for i in t:
                optimizer.minimize()
                log.coefficients(test_name, optimizer, signal_coeffs)
