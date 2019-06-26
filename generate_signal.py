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

# TODO: Check signal terms
# TODO: Switch to continuous PDF
# TODO: Should J transforms be used? (Based on q2 >> ...)
# TODO: Convert to tf distribution?
# TODO: Do fitting
# TODO: Split files/Unit tests/comments/doc comments

mass_mu = tf.constant(105.6583745e6)  # in 106 MeV/c^2
q2_min = tf.constant(1.0e18)  # 1 (GeV/c^2)^2
q2_max = tf.constant(6.0e18)  # 6 (GeV/c^2)^2
one = tf.constant(1.0)
two = tf.constant(2.0)
four = tf.constant(4.0)

# Outer arrays: [a_par_l, a_par_r, a_perp_l, a_perp_r, a_zero_l, a_zero_r]
# Inner arrays: [Re(...), Im(...)
# Inner array coeffs: [a, b, c] for anzatz a + (b * q2) + (c / q2)
signal_coeffs = [
    [[one, 0.0, 0.0], [two, 0.0, 0.0]],
    [[four, 0.0, 0.0], [one, 0.0, 0.0]],
    [[two, 0.0, 0.0], [four, 0.0, 0.0]],
    [[one, 0.0, 0.0], [two, 0.0, 0.0]],
    [[four, 0.0, 0.0], [one, 0.0, 0.0]],
    [[two, 0.0, 0.0], [four, 0.0, 0.0]],
]


def transform_amplitudes(original_amplitudes):
    [a_par_l, a_par_r, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = original_amplitudes

    # Original basis
    n_par = tf.Variable([a_par_l, tf.math.conj(a_par_r)])
    n_perp = tf.Variable([a_perp_l, - tf.math.conj(a_perp_r)])
    n_zero = tf.Variable([a_zero_l, tf.math.conj(a_zero_r)])

    # Transformation angle omega
    tan_2omega = two * (
            (tf.math.imag(a_zero_r) * tf.math.real(a_zero_l)) + (tf.math.imag(a_zero_l) * tf.math.real(a_zero_r))
    ) / (tf.math.abs(a_zero_r) ** 2 - tf.math.abs(a_zero_l) ** 2)
    omega = tf.math.atan(tan_2omega) / two
    tan_omega = tf.math.tan(omega)
    sinh_iomega = tf.cast(tf.math.sinh(tf.complex(0.0, omega)), tf.complex64)
    cosh_iomega = tf.cast(tf.math.cosh(tf.complex(0.0, omega)), tf.complex64)

    # Transformation angle theta
    tan_theta = (tf.math.real(a_zero_r) + (tf.math.imag(a_zero_l) * tan_omega)) / \
                (- tf.math.real(a_zero_l) + (tf.math.imag(a_zero_r) * tan_omega))
    theta = tf.math.atan(tan_theta)
    sin_theta = tf.cast(tf.math.sin(theta), tf.complex64)
    cos_theta = tf.cast(tf.math.cos(theta), tf.complex64)

    # Transformation angle phi_l
    tan_phi_l = (
            tf.math.imag(a_zero_l) +
            (tf.math.imag(a_zero_r) * tan_theta) -
            ((tf.math.real(a_zero_r) - (tf.math.real(a_zero_l) * tan_theta)) * tan_omega)
    ) / (
            - tf.math.real(a_zero_l) +
            (tf.math.real(a_zero_r) * tan_theta) +
            ((tf.math.imag(a_zero_r) + (tf.math.imag(a_zero_l) * tan_theta)) * tan_omega)
    )
    phi_l = tf.math.atan(tan_phi_l)
    exp_iphi_l = tf.cast(tf.math.exp(tf.complex(0.0, phi_l)), tf.complex64)

    # Transformation angle phi_r
    tan_phi_r = (
            tf.math.imag(a_perp_r) +
            (tf.math.imag(a_perp_l) * tan_theta) -
            ((tf.math.real(a_perp_l) - (tf.math.real(a_perp_r) * tan_theta)) * tan_omega)
    ) / (
            - tf.math.real(a_perp_r) +
            (tf.math.real(a_perp_l) * tan_theta) +
            ((tf.math.imag(a_perp_l) + (tf.math.imag(a_perp_r) * tan_theta)) * tan_omega)
    )
    phi_r = tf.math.atan(tan_phi_r)
    exp_miphi_r = tf.cast(tf.math.exp(tf.complex(0.0, - phi_r)), tf.complex64)

    # Transform basis
    def _transform_basis(n):
        # FIXME: Change this to operation on matrices directly
        return tf.Variable([
            exp_iphi_l * (
                    (cos_theta * ((n[0] * cosh_iomega) - n[1] * sinh_iomega)) -
                    (- sin_theta * ((- n[0] * sinh_iomega) + n[1] * cosh_iomega))
            ),
            exp_miphi_r * (
                    (sin_theta * ((n[0] * cosh_iomega) - n[1] * sinh_iomega)) -
                    (cos_theta * ((- n[0] * sinh_iomega) + n[1] * cosh_iomega))
            )
        ])
    nt_par = _transform_basis(n_par)
    nt_perp = _transform_basis(n_perp)
    nt_zero = _transform_basis(n_zero)

    # Transformed amplitudes (w/ basis fixing)
    a_par_l = nt_par[0]
    a_par_r = tf.math.conj(nt_par[1])
    a_perp_l = nt_perp[0]
    a_perp_r = tf.complex(- tf.math.real(nt_perp[1]), 0.0)
    a_zero_l = tf.complex(tf.math.real(nt_zero[0]), 0.0)
    a_zero_r = tf.constant(tf.complex(0.0, 0.0), shape=[a_zero_l.get_shape()[0]])

    return [a_par_l, a_par_r, a_perp_l, a_perp_r, a_zero_l, a_zero_r]


def decay_rate(independent_vars, amplitudes):
    q2 = independent_vars[:, 0]
    cos_theta_k = independent_vars[:, 1]
    cos_theta_l = independent_vars[:, 2]
    phi = independent_vars[:, 3]
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

    return (9 / (32 * math.pi)) * (
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


def generate_signal(signal_samples, options_num):
    q2_distribution = tfd.Uniform(low=q2_min, high=q2_max)
    cos_theta_k_distribution = tfd.Uniform(low=-1.0, high=1.0)
    cos_theta_l_distribution = tfd.Uniform(low=-1.0, high=1.0)
    phi_distribution = tfd.Uniform(low=-2*math.pi, high=2*math.pi)

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

    # Signal values for original amplitudes
    signal_amplitudes = []
    for amplitude_coeffs in signal_coeffs:
        components = []
        for component_coeffs in amplitude_coeffs:
            components.append(component_coeffs[0] + (component_coeffs[1] * q2) + (component_coeffs[2] / q2))
        signal_amplitudes.append(tf.complex(components[0], components[1]))
    _print("signal_amplitudes", signal_amplitudes)

    transformed_amplitudes = transform_amplitudes(signal_amplitudes)
    _print("transformed_amplitudes", transformed_amplitudes)

    decay_rates = decay_rate(options, transformed_amplitudes)
    _print("decay_rates", decay_rates)

    total_decay_rate = tf.reduce_sum(decay_rates)
    _print("total_decay_rate", total_decay_rate)

    probabilities = tf.math.maximum(decay_rates / total_decay_rate, 0.0)
    _print("probabilities", probabilities)

    keys = np.random.choice(options.get_shape()[0], signal_samples, p=probabilities.numpy())
    _print("keys", keys)

    signal = tf.gather(options, keys)
    _print("signal", signal)

    return signal


s = generate_signal(100_000, 10_000_000)

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Signal distributions')
titles = [
    r'$q^2$',
    r'$\cos{\theta_k}$',
    r'$\cos{\theta_l}$',
    r'$\phi$'
]

for ax, feature, title in zip(axes.flatten(), s.numpy().transpose(), titles):
    sns.distplot(feature, ax=ax, bins=20)
    ax.set(title=title)

plt.show()
