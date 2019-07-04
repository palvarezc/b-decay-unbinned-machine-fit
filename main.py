import matplotlib.pyplot as plt
import seaborn as sns

from signal_distribution import generate_events, pdf

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

# tf.debugging.set_log_device_placement(True)

# TODO: Fix basis fitting
# TODO: Fix fitting/Check one or two parameter fits
# TODO: Check maths terms
# TODO: Switch to accept-reject/monte-carlo. Increase sample size
# TODO: Add fitting graphs
# TODO: Do ensembles & plot distributions
# TODO: Optimise hyperparameters, optimiser, model
# TODO: Split files/Keras?/Convert to tf distribution?/Unit tests/method params/comments/doc comments

zero = tf.constant(0.0)

# Outer arrays: [a_par_l, a_par_r, a_perp_l, a_perp_r, a_zero_l, a_zero_r]
# Inner arrays: [Re(...), Im(...)
# Inner array coeffs: [a, b, c] for anzatz a + (b * q2) + (c / q2)
signal_coeffs = [
    [
        [-3.4277495848061257, -0.12410026985551571, 6.045281152442963],
        [0.00934061365013997, -0.001989193837745718, 0.5034113300277555]
    ],
    [
        [-0.25086978961912654, -0.005180213333933305, 8.636744983192575],
        [0.2220926359265556, -0.017419352926410284, -0.528067287659531]
    ],
    [
        [3.0646407176207813, 0.07851536717584778, -8.841144517240298],
        [-0.11366033229864046, 0.009293559978293, 0.04761546602270795]
    ],
    [
        [-0.9332669880450042, 0.01686711151445955, -6.318555350023665],
        [zero, zero, zero]
    ],
    [
        [5.882883042792871, -0.18442496620391777, 8.10139804649606],
        [zero, zero, zero]
    ],
    [
        [zero, zero, zero],
        [zero, zero, zero]
    ],
]


def build_coeff_structure(flat_coeffs):
    return [
        [flat_coeffs[0:3],   flat_coeffs[3:6]],
        [flat_coeffs[6:9],   flat_coeffs[9:12]],
        [flat_coeffs[12:15], flat_coeffs[15:18]],
        [flat_coeffs[18:21], [zero, zero, zero]],
        [flat_coeffs[21:24], [zero, zero, zero]],
        [[zero, zero, zero], [zero, zero, zero]],
    ]


def coeffs_to_string(coeffs):
    coeffs_strs = []
    for amp_coeffs in coeffs:
        for comp_coeffs in amp_coeffs:
            for coeff in comp_coeffs:
                if coeff is zero:
                    coeffs_strs.append(' *0.0*')
                else:
                    coeffs_strs.append('{:6.2f}'.format(coeff.numpy() if hasattr(coeff, 'numpy') else coeff))
    return ' '.join(coeffs_strs)


def nll(signal_events_, fit_coeffs_):
    coeff_struct_ = build_coeff_structure(fit_coeffs_)
    probs_ = pdf(signal_events_, coeff_struct_)
    nlog_probs = -tf.math.log(probs_)
    nll_ = -tf.reduce_sum(nlog_probs) / signal_events_.get_shape()[0]
    return nll_


#######################

signal_events = generate_events(100_000, signal_coeffs)

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

######################

fit_coeffs = [tf.Variable(1.0, name='c{0}'.format(i)) for i in range(24)]

# optimizer = tf.optimizers.Nadam(learning_rate=0.5)
optimizer = tf.optimizers.Adam(learning_rate=0.01)
# optimizer = tf.optimizers.SGD(learning_rate=0.01, momentum=0.01)

# import timeit
# print("nll x 1000: ", timeit.timeit(lambda: nll(signal_events, fit_coeffs), number=1000))
# exit(0)

print("Initial nll: {:.3f}".format(nll(signal_events, fit_coeffs)))

# Training loop
for i in range(10000):
    optimizer.minimize(
        lambda: nll(signal_events, fit_coeffs),
        var_list=fit_coeffs
    )
    if i % 20 == 0:
        print("Step {:03d}. nll: {:.3f}".format(i, nll(signal_events, fit_coeffs)))
        print("fit:    {}".format(coeffs_to_string(build_coeff_structure(fit_coeffs))))
        print("signal: {}".format(coeffs_to_string(signal_coeffs)))

