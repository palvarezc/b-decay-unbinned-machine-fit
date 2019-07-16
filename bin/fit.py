#!/usr/bin/env python
"""Fit amplitude coefficients to signal events"""

import matplotlib.pyplot as plt
import os
import seaborn as sns
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

# Append our found coefficients to this filename in the project folder
csv_filename = 'fit.csv'
csv_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', csv_filename)
# Whether to log for Tensorboard (Has large performance hit)
log = False
# Times to run
iterations = 10
# Restart iteration if we haven't converged after this many steps
step_restart = 20_000

with bmf.Script() as script:
    if log:
        log = bmf.Log(script.name)

    writer = bmf.CsvWriter(csv_filepath)
    if writer.current_id >= iterations:
        bmf.stdout('{} already contains {} iterations'.format(csv_filename, iterations))
        exit(0)

    while writer.current_id < iterations:
        bmf.stdout('Starting iteration {}/{}'.format(writer.current_id + 1, iterations))

        signal_coeffs = bmf.coeffs.signal()
        fit_coeffs = bmf.coeffs.fit()

        signal_events = bmf.signal.generate(signal_coeffs, events_total=2500)

        # Plot our signal distributions for each independent variable
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.suptitle('Signal distributions')
        titles = [
            r'$q^2$',
            r'$\cos{\theta_k}$',
            r'$\cos{\theta_l}$',
            r'$\phi$'
        ]
        for ax, feature, title in zip(axes.flatten(), signal_events.numpy().transpose(), titles):
            sns.distplot(feature, ax=ax, bins=20)
            ax.set(title=title)
        plt.show()

        optimizer = bmf.Optimizer(fit_coeffs, signal_events)

        def print_step():
            bmf.stdout(
                "Iteration:", "{}/{}". format(writer.current_id + 1, iterations),
                "Step:", optimizer.step,
                "Still training:", "{}/{}".format(optimizer.num_remaining(), len(optimizer.trainables)),
                "normalized_nll:", optimizer.normalized_nll,
            )
            bmf.stdout("fit:   ", bmf.coeffs.to_str(fit_coeffs))
            bmf.stdout("signal:", bmf.coeffs.to_str(signal_coeffs))

        print_step()

        while True:
            optimizer.minimize()
            if log:
                log.coefficients('fit_{}'.format(writer.current_id + 1), optimizer, signal_coeffs)
            if optimizer.converged():
                print_step()
                writer.write_coeffs(fit_coeffs)
                break
            if optimizer.step.numpy() % 100 == 0:
                print_step()
            if optimizer.step >= step_restart:
                bmf.stderr('No convergence after {} steps. Restarting iteration'.format(step_restart))
                break
