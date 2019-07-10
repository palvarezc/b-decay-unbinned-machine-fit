#!/usr/bin/env python
"""
Generate metrics for different optimizers and learning rates for comparison in Tensorboard

Once the script starts it will print out how to start Tensorboard, and the filter regex that can be
inputted in the left pane under 'Runs' to filter out just this run.
"""
import os
# This is bad form to put an assignment in the import section, but Tensorflow v2 will log without it
#  ruining our progress bars
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v2 as tf
from tensorflow.python.util import deprecation
from tqdm import trange

import b_meson_fit as bmf

# The log level change about should also disable deprecation warnings but it does not. Force them off
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.enable_v2_behavior()

# What we want to compare
optimizers = ['Adam', 'Nadam', 'SGD', 'RMSprop']
learning_rates = [0.005, 0.01, 0.05, 0.10, 0.15, 0.20]
iterations = 2000
params = {'optimizers': optimizers, 'learning_rates': learning_rates, 'iterations': iterations}

with bmf.Script(params=params, log=True) as script:
    signal_coeffs = bmf.coeffs.signal()
    signal_events = bmf.signal.generate(signal_coeffs)

    # Draw a signal line on each coefficient plot so we can compare how well the optimizers do
    bmf.Optimizer.log_signal_line(script, bmf.coeffs.fit(), signal_coeffs, iterations)

    for opt_name in optimizers:
        for lr in learning_rates:
            # Give this run the name `optimizer`-`learning_rate`
            script.log.suffix = "{}-{}".format(opt_name, lr)

            optimizer = bmf.Optimizer(
                script,
                bmf.coeffs.fit(),
                signal_events,
                opt_name,
                learning_rate=lr
            )

            try:
                # Use tqdm's trange() to print a progress bar for each optimizer/learning rate combo
                with trange(iterations, desc='{}/{}'.format(opt_name, lr)) as t:
                    for i in t:
                        grads = optimizer.minimize()
            except tf.errors.InvalidArgumentError:
                # Picking bad optimizer settings can result in a "underflow in dt" error from odeint()
                # Just quit this loop and carry on if that's the case
                script.stdout('Optimizer bailed. Continuing')
