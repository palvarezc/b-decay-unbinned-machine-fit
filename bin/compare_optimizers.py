#!/usr/bin/env python
"""
Generate metrics for different optimizers and learning rates for comparison in Tensorboard

Once the script starts it will print out how to start Tensorboard, and the filter regex that can be
inputted in the left pane under 'Runs' to filter out just this run.
"""

import datetime
import os
# This is bad form to put an assignment in the import section, but Tensorflow will log without it
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
learning_rates = [0.05, 0.01, 0.05, 0.10, 0.15, 0.20]
iterations = 1000

date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_top_dir = 'logs'
log_prefix = "optimizers/{}".format(date_str)
script_dir = os.path.dirname(os.path.realpath(__file__))

print('Starting comparision with settings:')
print(' * Optimizers: {}'.format(', '.join(optimizers)))
print(' * Learning rates: {}'.format(', '.join(map(str, learning_rates))))
print(' * Iterations: {}'.format(iterations))
print('')
print(
    'Start Tensorboard from the project folder with `tensorboard --logdir={}/ --host=127.0.0.1 --port=6006\'' +
    ' and navigate to http://127.0.0.1:6006'.format(log_top_dir)
)
print('Filter regex: {}'.format(log_prefix))
print('')

signal_events = bmf.signal.generate(bmf.coeffs.signal)

for opt in optimizers:
    for lr in learning_rates:
        # TODO: Improve portability of file path handling
        log_dir = "{}/../{}/{}-{}".format(script_dir, log_top_dir, log_prefix, opt, lr)
        summary_writer = tf.summary.create_file_writer(logdir=log_dir)

        with tf.device('/device:GPU:0'):
            nll = bmf.signal.nll(bmf.coeffs.signal, signal_events)
            optimizer = getattr(tf.optimizers, opt)(learning_rate=lr)

            # Use tqdm's trange() to print a progress bar for each optimizer/learning rate combo
            with trange(iterations, desc='{}/{}'.format(opt, lr)) as t:
                for i in t:
                    with tf.GradientTape() as tape:
                        nll = bmf.signal.nll(bmf.coeffs.fit, signal_events)
                    grads = tape.gradient(nll, bmf.coeffs.trainables())
                    optimizer.apply_gradients(zip(grads, bmf.coeffs.trainables()))

                    # Write all out Tensorboard stats
                    with summary_writer.as_default():
                        # Macro scalars
                        tf.summary.scalar('nll', nll, step=i)
                        tf.summary.scalar('gradients/max', tf.reduce_max(grads), step=i)
                        tf.summary.scalar('gradients/mean', tf.reduce_mean(grads), step=i)
                        tf.summary.scalar('gradients/total', tf.reduce_sum(grads), step=i)

                        # All trainable coefficients and gradients as individual scalars
                        for j in range(0, len(bmf.coeffs.trainables())):
                            coeff = bmf.coeffs.trainables()[j]
                            name = coeff.name.split(':')[0]
                            tf.summary.scalar('coefficients/' + name, coeff, step=i)
                            tf.summary.scalar('gradients/' + name, grads[j], step=i)

                        # Histogram data
                        tf.summary.histogram('gradients', grads, step=i)
                        tf.summary.histogram('coefficients', bmf.coeffs.fit, step=i)

                        # Ensure data is flushed to disk after each loop
                        tf.summary.flush()

print('')
print('Finished comparison.')