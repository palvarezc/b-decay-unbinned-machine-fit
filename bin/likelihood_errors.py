#!/usr/bin/env python
"""
Generate metrics for different optimizers and learning rates for comparison in Tensorboard
"""
import matplotlib
import tensorflow.compat.v2 as tf

import b_meson_fit as bmf

tf.enable_v2_behavior()

max_iterations = 50000
signal_model = bmf.coeffs.SM
signal_count = 20_000
step_size = 0.05
c_idx = 14

def fit(fit_coeffs_, signal_events_, learning_rate_):
    optimizer = bmf.Optimizer(
        fit_coeffs_,
        signal_events_,
        learning_rate=learning_rate_
    )

    for step in range(max_iterations):
        optimizer.minimize()
        if optimizer.converged():
            nll_ = optimizer.normalized_nll.numpy()  # * signal_count
            print('{} {} {} {}'.format(bmf.coeffs.names[c_idx], step, fit_coeffs_[c_idx].numpy(), nll_))
            return fit_coeffs_[c_idx].numpy(), nll_

with bmf.Script() as script:
    signal_coeffs = bmf.coeffs.signal(signal_model)

    signal_events = bmf.signal.generate(signal_coeffs, signal_count)
    fit_coeffs = bmf.coeffs.fit(bmf.coeffs.FIT_INIT_CURRENT_SIGNAL, signal_model)

    x_list = []
    y_list = []

    val, nll = fit(fit_coeffs, signal_events, 0.005)
    x_list.append(val)
    y_list.append(nll)

    for step_sign in [-1, +1]:
        fit_coeffs2 = []
        for coeff in fit_coeffs:
            if bmf.coeffs.is_trainable(coeff):
                fit_coeffs2.append(tf.Variable(coeff.numpy()))
            else:
                fit_coeffs2.append(tf.constant(coeff.numpy()))

        for step in range(1, 11):
            fit_coeffs2[c_idx] = tf.constant(
                fit_coeffs[c_idx].numpy() + (step * step_size * step_sign),
                dtype=tf.float32
            )
            bmf.stdout(bmf.coeffs.to_str(fit_coeffs2))
            val, nll = fit(fit_coeffs2, signal_events, 0.001)
            x_list.append(val)
            y_list.append(nll)

            # if len(y_list) > 0 and nll >= y_list[0] + 0.1:
            #     break

    print(x_list)
    print(y_list)
    lists = zip(x_list, y_list)
    y_list = [x for _, x in sorted(lists)]
    x_list = sorted(x_list)
    print(x_list)
    print(y_list)

    # Import these after we optionally set SVG backend - otherwise matplotlib may bail on a missing TK backend when
    #  running from the CLI
    import matplotlib.pylab as plt
    plt.rcParams.update({'figure.max_open_warning': 0})
    import seaborn as sns

    plt.figure()
    # Set style as well as font to Computer Modern Roman to match LaTeX output
    sns.set(style='ticks', font='cmr10', rc={'mathtext.fontset': 'cm', 'axes.unicode_minus': False})

    plt.title(bmf.coeffs.latex_names[c_idx])
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

    plt.plot(x_list, y_list)

    plt.show()
