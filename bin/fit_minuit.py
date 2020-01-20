#!/usr/bin/env python
"""Fit amplitude coefficients to signal events"""

import argparse
import shutil
import tensorflow.compat.v2 as tf
import tqdm
from iminuit import Minuit

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
    choices=bmf.coeffs.signal_models,
    default=bmf.coeffs.SM,
    help='signal model (default: {})'.format(bmf.coeffs.SM)
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
parser.add_argument(
        '-g',
            '--generator-seed',
            dest='seed',
            type=int,
            default=-1,
            help='seed for the generator (default: 1)'
        )
args = parser.parse_args()

if (not args.seed==-1): tf.random.set_seed(args.seed)

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

    if args.csv_file is not None:
        writer = bmf.FitWriter(args.csv_file, signal_coeffs)
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

        signal_events = bmf.signal.generate(signal_coeffs, events_total=args.signal_count)


        fit_coeffs = bmf.coeffs.fit(args.fit_init, args.signal_model)


        optimizer = bmf.Optimizer(
            fit_coeffs,
            signal_events,
            opt_name=args.opt_name,
            learning_rate=args.learning_rate,
            opt_params=opt_params,
            grad_clip=args.grad_clip,
            grad_max_cutoff=args.grad_max_cutoff
            )



        fit_vars = []
        for coeff in fit_coeffs:
            if type(coeff)==type(fit_coeffs[0]): fit_vars.append(coeff)

        pars_init = []
        for coeff in fit_vars: pars_init.append(coeff.numpy())

        # feed_dict_ = {}
        # for coeff in fit_coeffs: feed_dict_[coeff[0:-2]] = 0

        def func(pars):
            j=0
            for i in range(len(fit_coeffs)):
                if type(fit_coeffs[i])==type(fit_coeffs[0]):
                    fit_coeffs[i].assign(pars[j])
                    j+=1

            return optimizer.normalized_nll_feed(fit_coeffs).numpy()

        m = Minuit.from_array_func(func, pars_init)
        m.migrad()
        m.hesse()
        # m.minos()
                            

        if args.log:
            log.coefficients('fit_{}'.format(iteration), optimizer, signal_coeffs)

        if args.csv_file is not None:
            pars_final = m.values.values()
            j=0
            for i in range(len(fit_coeffs)): 
                if type(fit_coeffs[i])==type(fit_coeffs[0]):
                    fit_coeffs[i].assign(pars_final[j])
                    j+=1
            writer.write_coeffs(func(pars_final), fit_coeffs, script.timer_elapsed('fit'))





            
