#!/usr/bin/env python
"""Fit amplitude coefficients to signal events"""

import argparse
import os
import shutil
import tensorflow.compat.v2 as tf
import tqdm

import b_meson_fit as bmf

# Only do plots if running PyCharm
if 'PYCHARM_HOSTED' in os.environ:
    import matplotlib.pylab as plt
    import seaborn as sns

tf.enable_v2_behavior()

columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Fit coefficients to generated toy signal(s).',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=columns, width=columns),
)
parser.add_argument(
    '-c',
    '--csv',
    dest='csv_file',
    help='write results to this CSV file'
)
parser.add_argument(
    '-d',
    '--device',
    dest='device',
    default=bmf.Script.device_default,
    help='use this device e.g. CPU:0, GPU:0, GPU:1 (default: {})'.format(bmf.Script.device_default),
)
parser.add_argument(
    '-i',
    '--iterations',
    dest='iterations',
    type=int,
    default=1,
    help='number of iterations to run (default: 1)'
)
parser.add_argument(
    '-l',
    '--log',
    dest='log',
    action='store_true',
    help='store logs for Tensorboard (has large performance hit)'
)
parser.add_argument(
    '-m',
    '--max-step',
    dest='max_step',
    type=int,
    default=20_000,
    help='restart iteration if not converged after this many steps (default: 20000)'
)
parser.add_argument(
    '-o',
    '--opt-name',
    dest='opt_name',
    default=bmf.Optimizer.opt_name_default,
    help='optimizer algorithm to use (default: {})'.format(bmf.Optimizer.opt_name_default),
)
parser.add_argument(
    '-p',
    '--opt-param',
    nargs=2,
    dest='opt_params',
    action='append',
    metavar=('PARAM_NAME', 'VALUE'),
    help='additional params to pass to optimizer - can be specified multiple times'
)
parser.add_argument(
    '-P',
    '--grad-clip',
    dest='grad_clip',
    type=float,
    help='clip gradients by this global norm'
)
parser.add_argument(
    '-r',
    '--learning-rate',
    dest='learning_rate',
    type=float,
    default=bmf.Optimizer.learning_rate_default,
    help='optimizer learning rate (default: {})'.format(bmf.Optimizer.learning_rate_default),
)
parser.add_argument(
    '-s',
    '--signal-count',
    dest='signal_count',
    type=int,
    default=2400,
    help='number of signal events to generated per fit (default: 2400)'
)
parser.add_argument(
    '-u',
    '--grad-max-cutoff',
    dest='grad_max_cutoff',
    type=float,
    default=bmf.Optimizer.grad_max_cutoff_default,
    help='count fit as converged when max gradient is less than this ' +
         '(default: {})'.format(bmf.Optimizer.grad_max_cutoff_default),
)
args = parser.parse_args()

# Convert optimizer params to dict
opt_params = {}
if args.opt_params:
    for idx, _ in enumerate(args.opt_params):
        # Change any opt param values to floats if possible
        try:
            args.opt_params[idx][1] = float(args.opt_params[idx][1])
        except ValueError:
            pass
        opt_params[args.opt_params[idx][0]] = args.opt_params[idx][1]

iteration = 0
with bmf.Script(device=args.device) as script:
    if args.log:
        log = bmf.Log(script.name)

    if args.csv_file is not None:
        writer = bmf.CsvWriter(args.csv_file)
        if writer.current_id > 0:
            bmf.stdout('{} already contains {} iteration(s)'.format(args.csv_file, writer.current_id))
            bmf.stdout('')
            if writer.current_id >= args.iterations:
                bmf.stderr('Nothing to do')
                exit(0)
            iteration = writer.current_id

    # Show progress bar for fits
    for iteration in tqdm.trange(
            iteration + 1,
            args.iterations + 1,
            initial=iteration,
            total=args.iterations,
            unit='fit'
    ):
        # Time each iteration for CSV writing
        script.timer_start('fit')

        signal_coeffs = bmf.coeffs.signal()
        signal_events = bmf.signal.generate(signal_coeffs, events_total=args.signal_count)

        # If running if PyCharm, plot our signal distributions for each independent variable
        if 'PYCHARM_HOSTED' in os.environ:
            fig, axes = plt.subplots(nrows=2, ncols=2)
            fig.suptitle('Signal (Iteration {}/{})'.format(iteration, args.iterations))
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

        attempt = 1
        converged = False
        while not converged:
            fit_coeffs = bmf.coeffs.fit()
            optimizer = bmf.Optimizer(
                fit_coeffs,
                signal_events,
                opt_name=args.opt_name,
                learning_rate=args.learning_rate,
                opt_params=opt_params,
                grad_clip=args.grad_clip,
                grad_max_cutoff=args.grad_max_cutoff
            )

            while True:
                optimizer.minimize()
                if args.log:
                    log.coefficients('fit_{}'.format(iteration), optimizer, signal_coeffs)
                if optimizer.converged():
                    converged = True
                    if args.csv_file is not None:
                        writer.write_coeffs(optimizer.normalized_nll, fit_coeffs, script.timer_elapsed('fit'))
                    break
                if optimizer.step >= args.max_step:
                    bmf.stderr('No convergence after {} steps. Restarting iteration'.format(args.max_step))
                    attempt = attempt + 1
                    break
