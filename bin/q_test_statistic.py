#!/usr/bin/env python
"""Generate Q test statistics"""

import argparse
import os
import shutil
import tensorflow.compat.v2 as tf
import tqdm

import b_meson_fit as bmf

tf.enable_v2_behavior()


def fit_init_value(arg):  # Handle --fit-init argument
    if arg in bmf.coeffs.fit_init_schemes:
        return arg
    try:
        init_value = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(
            '{} is not one of {}'.format(arg, ",".join(bmf.coeffs.fit_init_schemes + ['FLOAT']))
        )
    return init_value


columns = shutil.get_terminal_size().columns
parser = argparse.ArgumentParser(
    description='Generate Q test statistics for a signal model.',
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=columns, width=columns),
)
parser.add_argument(
    '-d',
    '--device',
    dest='device',
    default=bmf.Script.device_default,
    help='use this device e.g. CPU:0, GPU:0, GPU:1 (default: {})'.format(bmf.Script.device_default),
)
parser.add_argument(
    '-f',
    '--fit-init',
    dest='fit_init',
    type=fit_init_value,
    metavar='FIT_INIT',
    default=bmf.coeffs.fit_initialization_scheme_default,
    help='fit coefficient initialization. FIT_INIT should be one of {} (default: {})'.format(
        bmf.coeffs.fit_init_schemes + ['FLOAT'],
        bmf.coeffs.fit_initialization_scheme_default
    )
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
    '-n',
    '--null-model',
    dest='null_model',
    required=True,
    choices=bmf.coeffs.signal_models,
    help='fix null P-wave fit coeffs to the values in this signal model'
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
    '-S',
    '--signal-model',
    dest='signal_model',
    required=True,
    choices=bmf.coeffs.signal_models,
    default=bmf.coeffs.SM,
    help='signal model (default: {})'.format(bmf.coeffs.SM)
)
parser.add_argument(
    '-t',
    '--test-model',
    dest='test_model',
    required=True,
    choices=bmf.coeffs.signal_models,
    help='fix test P-wave fit coeffs to the values in this signal model'
)
parser.add_argument(
    '-x',
    '--txt',
    dest='txt_file',
    required=True,
    help='write results to this txt file'
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

    signal_coeffs = bmf.coeffs.signal(args.signal_model)

    if os.path.isfile(args.txt_file):
        num_lines = sum(1 for line in open(args.txt_file))
        bmf.stdout('{} already contains {} iteration(s)'.format(args.txt_file, num_lines))
        bmf.stdout('')
        if num_lines >= args.iterations:
            bmf.stderr('Nothing to do')
            exit(0)
        iteration = num_lines

    with open(args.txt_file, 'a') as txt_file:
        # Show progress bar for statistics
        for iteration in tqdm.trange(
                iteration + 1,
                args.iterations + 1,
                initial=iteration,
                total=args.iterations,
                unit='test stat'
        ):
            signal_events = bmf.signal.generate(signal_coeffs, events_total=args.signal_count)

            def nll(fix_coeffs_model):
                attempt = 1
                while True:
                    fit_coeffs = bmf.coeffs.fit(args.fit_init, args.signal_model, fix_coeffs_model)
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
                            log.coefficients('q_test_stat_{}'.format(iteration), optimizer, signal_coeffs)
                        if optimizer.converged():
                            # Multiply the normalized_nll up into the full nll
                            return optimizer.normalized_nll * args.signal_count
                        if optimizer.step >= args.max_step:
                            bmf.stderr('No convergence after {} steps. Restarting iteration'.format(args.max_step))
                            attempt = attempt + 1
                            break

            test_nll = nll(args.test_model)
            null_nll = nll(args.null_model)
            q = 2 * (test_nll - null_nll)

            txt_file.write('{}\n'.format(q))
