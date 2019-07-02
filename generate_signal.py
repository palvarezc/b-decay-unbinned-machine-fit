import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import tensorflow_probability as tfp

from tensorflow.python import tf2
if not tf2.enabled():
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions

# TODO: Do fitting
# TODO: Fix basis fitting

# TODO: n_sig?/normalisation/d_avg_dat?
# TODO: Check PDF terms
# TODO: Think about floor functions
# TODO: https://www.tensorflow.org/beta/guide/autograph
# TODO: Switch to accept-reject/monte-carlo. Increase sample size
# TODO: Add fitting graphs
# TODO: Normalise variables?
# TODO: Do ensambles & plot distributions
# TODO: Use Keras?
# TODO: Optimise hyperparameters, optimiser, model
# TODO: Split files/Convert to tf distribution?/Unit tests/method params/comments/doc comments

mass_mu = tf.constant(0.1056583745)  # in 0.106 GeV/c^2
q2_min = tf.constant(1.0)  # 1 (GeV/c^2)^2
q2_max = tf.constant(8.0)  # 8 (GeV/c^2)^2
zero = tf.constant(0.0)
one = tf.constant(1.0)
two = tf.constant(2.0)
four = tf.constant(4.0)

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


# def transform_amplitudes(original_amplitudes):
#     [a_par_l, a_par_r, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = original_amplitudes
#
#     # Original basis
#     n_par = tf.Variable([a_par_l, tf.math.conj(a_par_r)])
#     n_perp = tf.Variable([a_perp_l, - tf.math.conj(a_perp_r)])
#     n_zero = tf.Variable([a_zero_l, tf.math.conj(a_zero_r)])
#
#     # Transformation angle omega
#     tan_2omega = two * (
#             (tf.math.imag(a_zero_r) * tf.math.real(a_zero_l)) + (tf.math.imag(a_zero_l) * tf.math.real(a_zero_r))
#     ) / (tf.math.abs(a_zero_r) ** 2 - tf.math.abs(a_zero_l) ** 2)
#     omega = tf.math.atan(tan_2omega) / two
#     tan_omega = tf.math.tan(omega)
#     sinh_iomega = tf.cast(tf.math.sinh(tf.complex(0.0, omega)), tf.complex64)
#     cosh_iomega = tf.cast(tf.math.cosh(tf.complex(0.0, omega)), tf.complex64)
#
#     # Transformation angle theta
#     tan_theta = (tf.math.real(a_zero_r) + (tf.math.imag(a_zero_l) * tan_omega)) / \
#                 (- tf.math.real(a_zero_l) + (tf.math.imag(a_zero_r) * tan_omega))
#     theta = tf.math.atan(tan_theta)
#     sin_theta = tf.cast(tf.math.sin(theta), tf.complex64)
#     cos_theta = tf.cast(tf.math.cos(theta), tf.complex64)
#
#     # Transformation angle phi_l
#     tan_phi_l = (
#             tf.math.imag(a_zero_l) +
#             (tf.math.imag(a_zero_r) * tan_theta) -
#             ((tf.math.real(a_zero_r) - (tf.math.real(a_zero_l) * tan_theta)) * tan_omega)
#     ) / (
#             - tf.math.real(a_zero_l) +
#             (tf.math.real(a_zero_r) * tan_theta) +
#             ((tf.math.imag(a_zero_r) + (tf.math.imag(a_zero_l) * tan_theta)) * tan_omega)
#     )
#     phi_l = tf.math.atan(tan_phi_l)
#     exp_iphi_l = tf.cast(tf.math.exp(tf.complex(0.0, phi_l)), tf.complex64)
#
#     # Transformation angle phi_r
#     tan_phi_r = (
#             tf.math.imag(a_perp_r) +
#             (tf.math.imag(a_perp_l) * tan_theta) -
#             ((tf.math.real(a_perp_l) - (tf.math.real(a_perp_r) * tan_theta)) * tan_omega)
#     ) / (
#             - tf.math.real(a_perp_r) +
#             (tf.math.real(a_perp_l) * tan_theta) +
#             ((tf.math.imag(a_perp_l) + (tf.math.imag(a_perp_r) * tan_theta)) * tan_omega)
#     )
#     phi_r = tf.math.atan(tan_phi_r)
#     exp_miphi_r = tf.cast(tf.math.exp(tf.complex(0.0, - phi_r)), tf.complex64)
#
#     # Transform basis
#     def _transform_basis(n):
#         # FIXME: Change this to operation on matrices directly
#         return tf.Variable([
#             exp_iphi_l * (
#                     (cos_theta * ((n[0] * cosh_iomega) - n[1] * sinh_iomega)) -
#                     (- sin_theta * ((- n[0] * sinh_iomega) + n[1] * cosh_iomega))
#             ),
#             exp_miphi_r * (
#                     (sin_theta * ((n[0] * cosh_iomega) - n[1] * sinh_iomega)) -
#                     (cos_theta * ((- n[0] * sinh_iomega) + n[1] * cosh_iomega))
#             )
#         ])
#     nt_par = _transform_basis(n_par)
#     nt_perp = _transform_basis(n_perp)
#     nt_zero = _transform_basis(n_zero)
#
#     # Transformed amplitudes (w/ basis fixing)
#     a_par_l = nt_par[0]
#     a_par_r = tf.math.conj(nt_par[1])
#     a_perp_l = nt_perp[0]
#     a_perp_r = tf.complex(- tf.math.real(nt_perp[1]), 0.0)
#     a_zero_l = tf.complex(tf.math.real(nt_zero[0]), 0.0)
#     a_zero_r = tf.constant(tf.complex(0.0, 0.0), shape=[a_zero_l.get_shape()[0]])
#
#     return [a_par_l, a_par_r, a_perp_l, a_perp_r, a_zero_l, a_zero_r]


def decay_rate(q2, cos_theta_k, cos_theta_l, phi, amplitudes):
    [a_par_l, a_par_r, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = amplitudes

    # Angles
    cos2_theta_k = cos_theta_k**2
    sin2_theta_k = one - cos2_theta_k
    sin_theta_k = tf.sqrt(sin2_theta_k)
    sin_2theta_k = two * sin_theta_k * cos_theta_k

    cos2_theta_l = cos_theta_l**2
    cos_2theta_l = (two * cos2_theta_l) - one
    sin2_theta_l = one - cos2_theta_l
    sin_theta_l = tf.sqrt(sin2_theta_l)
    sin_2theta_l = two * sin_theta_l * cos_theta_l

    cos_phi = tf.math.cos(phi)
    cos_2phi = tf.math.cos(two * phi)
    sin_phi = tf.math.sin(phi)
    sin_2phi = tf.math.sin(two * phi)

    # Mass terms
    four_mass_mu_over_q2 = (four * (mass_mu**2)) / q2
    beta2_mu = one - four_mass_mu_over_q2
    beta_mu = tf.sqrt(beta2_mu)

    # Angular observables
    j_1s = ((two + beta2_mu) / four)*(
        tf.math.abs(a_perp_l)**2 + tf.math.abs(a_par_l)**2 +
        tf.math.abs(a_perp_r)**2 + tf.math.abs(a_par_r)**2
    ) + four_mass_mu_over_q2*tf.math.real(
        a_perp_l * tf.math.conj(a_perp_r) +
        a_par_l * tf.math.conj(a_par_r)
    )

    j_1c = tf.math.abs(a_zero_l)**2 + tf.math.abs(a_zero_r)**2 + \
        four_mass_mu_over_q2*(2*tf.math.real(a_zero_l * tf.math.conj(a_zero_r)))

    j_2s = (beta2_mu / four)*(
        tf.math.abs(a_perp_l)**2 + tf.math.abs(a_par_l)**2 +
        tf.math.abs(a_perp_r)**2 + tf.math.abs(a_par_r)**2
    )

    j_2c = (- beta2_mu)*(tf.math.abs(a_zero_l)**2 + tf.math.abs(a_zero_r)**2)

    j_3 = (beta2_mu / two) * (
        tf.math.abs(a_perp_l) ** 2 - tf.math.abs(a_par_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 - tf.math.abs(a_par_r) ** 2
    )

    j_4 = (beta2_mu / tf.sqrt(two)) * (
        tf.math.real(a_zero_l * tf.math.conj(a_par_l)) +
        tf.math.real(a_zero_r * tf.math.conj(a_par_r))
    )

    j_5 = tf.sqrt(two) * beta_mu * (
        tf.math.real(a_zero_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_zero_r * tf.math.conj(a_perp_r))
    )

    j_6s = two * beta_mu * (
        tf.math.real(a_par_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_par_r * tf.math.conj(a_perp_r))
    )

    j_7 = tf.sqrt(two) * beta_mu * (
        tf.math.imag(a_zero_l * tf.math.conj(a_par_l)) -
        tf.math.imag(a_zero_r * tf.math.conj(a_par_r))
    )

    j_8 = (beta2_mu / tf.sqrt(two)) * (
        tf.math.imag(a_zero_l * tf.math.conj(a_perp_l)) +
        tf.math.imag(a_zero_r * tf.math.conj(a_perp_r))
    )

    j_9 = beta2_mu * (
        tf.math.imag(tf.math.conj(a_par_l) * a_perp_l) +
        tf.math.imag(tf.math.conj(a_par_r) * a_perp_r)
    )

    observables = tf.stack([
        j_1s,
        j_1c,
        j_2s,
        j_2c,
        j_3,
        j_4,
        j_5,
        j_6s,
        j_7,
        j_8,
        j_9
    ])

    rate = (9 / (32 * math.pi)) * (
        (j_1s * sin2_theta_k) +
        (j_1c * cos2_theta_k) +
        (j_2s * sin2_theta_k * cos_2theta_l) +
        (j_2c * cos2_theta_k * cos_2theta_l) +
        (j_3 * sin2_theta_k * sin2_theta_l * cos_2phi) +
        (j_4 * sin_2theta_k * sin_2theta_l * cos_phi) +
        (j_5 * sin_2theta_k * sin_theta_l * cos_phi) +
        (j_6s * sin2_theta_k * cos_theta_l) +
        (j_7 * sin_2theta_k * sin_theta_l * sin_phi) +
        (j_8 * sin_2theta_k * sin_2theta_l * sin_phi) +
        (j_9 * sin_2theta_k * sin_2theta_l * sin_2phi)
    )

    return [observables, rate]


def coeffs_to_amplitudes(q2, all_coeffs):
    amplitudes = []
    for amplitude_coeffs in all_coeffs:
        components = []
        for component_coeffs in amplitude_coeffs:
            components.append(component_coeffs[0] + (component_coeffs[1] * q2) + (component_coeffs[2] / q2))
        amplitudes.append(tf.complex(components[0], components[1]))

    return amplitudes


def generate_signal(signal_samples, options_num):
    q2_distribution = tfd.Uniform(low=q2_min, high=q2_max)
    cos_theta_k_distribution = tfd.Uniform(low=-1.0, high=1.0)
    cos_theta_l_distribution = tfd.Uniform(low=-1.0, high=1.0)
    phi_distribution = tfd.Uniform(low=-math.pi, high=math.pi)

    def _print(name, t):
        tf.print(name, "(shape:", tf.shape(t), "type:", type(t), "):\n", t, output_stream=sys.stdout, end="\n\n")

    q2 = q2_distribution.sample(options_num)
    options = tf.stack(
        [
            q2,
            cos_theta_k_distribution.sample(options_num),
            cos_theta_l_distribution.sample(options_num),
            phi_distribution.sample(options_num)
        ],
        axis=1,
        name='signal_options'
    )
    _print("options", options)

    _print("signal_coeffs", signal_coeffs)

    # Signal values for original amplitudes
    original_amplitudes = coeffs_to_amplitudes(q2, signal_coeffs)
    _print("signal_amplitudes", original_amplitudes)

    #transformed_amplitudes = transform_amplitudes(original_amplitudes)
    transformed_amplitudes = original_amplitudes
    _print("transformed_amplitudes", transformed_amplitudes)

    [observables, decay_rates] = \
        decay_rate(options[:, 0], options[:, 1], options[:, 2], options[:, 3], transformed_amplitudes)
    _print("observables", observables)
    _print("decay_rates", decay_rates)

    total_decay_rate = tf.reduce_sum(decay_rates)
    _print("total_decay_rate", total_decay_rate)

    probabilities = tf.math.maximum(decay_rates / total_decay_rate, 0.0)
    _print("probabilities", probabilities)

    keys = np.random.choice(options.get_shape()[0], signal_samples, p=probabilities.numpy())
    _print("keys", keys)

    signal = tf.gather(options, keys)
    _print("signal", signal)

    signal_decay_rates = tf.gather(decay_rates, keys)
    _print("signal_decay_rates", signal_decay_rates)

    signal_observables = tf.gather(observables, keys, axis=1)
    _print("signal_observables", signal_observables)

    return [signal, signal_decay_rates, signal_observables]


def build_coeff_structure(flat_coeffs):
    return [
        [flat_coeffs[0:3],   flat_coeffs[3:6]],
        [flat_coeffs[6:9],   flat_coeffs[9:12]],
        [flat_coeffs[12:15], flat_coeffs[15:18]],
        [flat_coeffs[18:21], [zero, zero, zero]],
        [flat_coeffs[21:24], [zero, zero, zero]],
        [[zero, zero, zero], [zero, zero, zero]],
    ]


def generate_attempt(independent_vars_, amplitudes):
    attempt_coeffs = build_coeff_structure(amplitudes)
    transformed_amplitudes = coeffs_to_amplitudes(independent_vars_[:, 0], attempt_coeffs)

    [observables, decay_rates] = decay_rate(
        independent_vars_[:, 0],
        independent_vars_[:, 1],
        independent_vars_[:, 2],
        independent_vars_[:, 3],
        transformed_amplitudes
    )

    return [decay_rates, observables]


def nll(independent_vars_, fit_coeffs_):
    [fit_decay_rates_, _] = generate_attempt(independent_vars_, fit_coeffs_)
    v1 = tf.math.maximum(fit_decay_rates_, 1e-20)
    v2 = -tf.math.log(v1)
    v3 = -tf.reduce_sum(v2) / independent_vars_.get_shape()[0]
    return v3


def coeffs_to_string(coeffs):
    coeffs_strs = []
    for amp_coeffs in coeffs:
        for comp_coeffs in amp_coeffs:
            for coeff in comp_coeffs:
                if coeff is zero:
                    coeffs_strs.append(' *0.0*')
                else:
                    coeffs_strs.append('{:5.2f}'.format(coeff.numpy() if hasattr(coeff, 'numpy') else coeff))
    return ' '.join(coeffs_strs)


#######################

[independent_vars, _, _] = generate_signal(100_000, 10_000_000)

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Signal distributions')
titles = [
    r'$q^2$',
    r'$\cos{\theta_k}$',
    r'$\cos{\theta_l}$',
    r'$\phi$'
]

for ax, feature, title in zip(axes.flatten(), independent_vars.numpy().transpose(), titles):
    sns.distplot(feature, ax=ax, bins=20)
    ax.set(title=title)

plt.show()

######################

fit_coeffs = [tf.Variable(1.0, name='c{0}'.format(i)) for i in range(24)]

optimizer = tf.optimizers.Nadam()
# optimizer = tf.optimizers.Adam(learning_rate=0.01)
# optimizer = tf.optimizers.SGD(learning_rate=0.1)

print("Initial loss: {:.3f}".format(nll(independent_vars, fit_coeffs)))

# Training loop
for i in range(200):
    optimizer.minimize(
        lambda: nll(independent_vars, fit_coeffs),
        var_list=fit_coeffs
    )
    if i % 20 == 0:
        print(
            "Step {:03d}. nll: {:.3f} fit_coeffs: {}".format(
                i,
                nll(independent_vars, fit_coeffs),
                coeffs_to_string(build_coeff_structure(fit_coeffs))
            )
        )
        print("                  signal_coeffs: {}".format(coeffs_to_string(signal_coeffs)))

