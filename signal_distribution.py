import math
import numpy as np
import tensorflow_probability as tfp

# Import this separately as its old Tensorflow v1 code
from tensorflow.contrib import integrate as tf_integrate

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

q2_min = tf.constant(1.0)  # 1 (GeV/c^2)^2
q2_max = tf.constant(8.0)  # 8 (GeV/c^2)^2
mass_mu = tf.constant(0.1056583745)  # in 0.106 GeV/c^2
one = tf.constant(1.0)
two = tf.constant(2.0)
four = tf.constant(4.0)
options_num = 10_000_000


# A @tf.function decoration makes this 3-3.5x faster, however as odeint() is not Tensorflow v2
#  compatible, after 1000s of iterations the gradients can all come randomly back as NaN. Very sad.
#  This error manifests itself as odeint complaining of a float (dt) underflow.
#  If odeint() is ported to tf_scientific then remove the decoration from child functions and add it here.
def pdf(signal_events_, coeffs):
    # Forcing the GPU here is about 25% faster. Might no longer be needed with @tf.function decoration
    with tf.device('/device:GPU:0'):
        decay_rates = _decay_rate(signal_events_, coeffs)
        norm = _integrate_decay_rate(coeffs)
    return tf.math.maximum(decay_rates / norm, 1e-30)


def generate_events(events_num, coeffs):
    q2 = tfp.distributions.Uniform(low=q2_min, high=q2_max).sample(options_num)
    cos_theta_k = tfp.distributions.Uniform(low=-1.0, high=1.0).sample(options_num)
    cos_theta_l = tfp.distributions.Uniform(low=-1.0, high=1.0).sample(options_num)
    phi = tfp.distributions.Uniform(low=-math.pi, high=math.pi).sample(options_num)

    options = tf.stack([q2, cos_theta_k, cos_theta_l, phi], axis=1, name='signal_options')

    probs = pdf(options, coeffs)
    normalised_probs = pdf(options, coeffs) / tf.reduce_sum(probs)
    choices = np.random.choice(options.get_shape()[0], events_num, p=normalised_probs.numpy())

    return tf.gather(options, choices)


@tf.function
def _decay_rate(signal_events_, coeffs):
    [q2, cos_theta_k, cos_theta_l, phi] = tf.unstack(signal_events_, axis=1)
    amplitudes = _coeffs_to_amplitudes(q2, coeffs)

    # Angles
    cos2_theta_k = cos_theta_k ** 2
    sin2_theta_k = one - cos2_theta_k
    sin_theta_k = tf.sqrt(sin2_theta_k)
    sin_2theta_k = two * sin_theta_k * cos_theta_k

    cos2_theta_l = cos_theta_l ** 2
    cos_2theta_l = (two * cos2_theta_l) - one
    sin2_theta_l = one - cos2_theta_l
    sin_theta_l = tf.sqrt(sin2_theta_l)
    sin_2theta_l = two * sin_theta_l * cos_theta_l

    cos_phi = tf.math.cos(phi)
    cos_2phi = tf.math.cos(two * phi)
    sin_phi = tf.math.sin(phi)
    sin_2phi = tf.math.sin(two * phi)

    # Mass terms
    four_mass_mu_over_q2 = (four * (mass_mu ** 2)) / q2
    beta2_mu = one - four_mass_mu_over_q2
    beta_mu = tf.sqrt(beta2_mu)

    return (9 / (32 * math.pi)) * (
        (_j1s(amplitudes, beta2_mu,  four_mass_mu_over_q2) * sin2_theta_k) +
        (_j1c(amplitudes, four_mass_mu_over_q2) * cos2_theta_k) +
        (_j2s(amplitudes, beta2_mu) * sin2_theta_k * cos_2theta_l) +
        (_j2c(amplitudes, beta2_mu) * cos2_theta_k * cos_2theta_l) +
        (_j3(amplitudes, beta2_mu) * sin2_theta_k * sin2_theta_l * cos_2phi) +
        (_j4(amplitudes, beta2_mu) * sin_2theta_k * sin_2theta_l * cos_phi) +
        (_j5(amplitudes, beta_mu) * sin_2theta_k * sin_theta_l * cos_phi) +
        (_j6s(amplitudes, beta_mu) * sin2_theta_k * cos_theta_l) +
        (_j7(amplitudes, beta_mu) * sin_2theta_k * sin_theta_l * sin_phi) +
        (_j8(amplitudes, beta2_mu) * sin_2theta_k * sin_2theta_l * sin_phi) +
        (_j9(amplitudes, beta2_mu) * sin_2theta_k * sin_2theta_l * sin_2phi)
    )


# https://arxiv.org/abs/1202.4266
# @see notebook
@tf.function
def _decay_rate_angle_integrated(q2, coeffs):
    amplitudes = _coeffs_to_amplitudes(q2, coeffs)

    # Mass terms
    four_mass_mu_over_q2 = (four * (mass_mu ** 2)) / q2
    beta2_mu = one - four_mass_mu_over_q2

    return (1 / 4) * (
        (6 * _j1s(amplitudes, beta2_mu,  four_mass_mu_over_q2)) +
        (3 * _j1c(amplitudes, four_mass_mu_over_q2)) -
        (2 * _j2s(amplitudes, beta2_mu)) -
        _j2c(amplitudes, beta2_mu)
    )


def _j1s(amplitudes, beta2_mu, four_mass_mu_over_q2):
    [a_par_l, a_par_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return ((two + beta2_mu) / four) * (
        tf.math.abs(a_perp_l) ** 2 + tf.math.abs(a_par_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 + tf.math.abs(a_par_r) ** 2
    ) + four_mass_mu_over_q2 * tf.math.real(
        a_perp_l * tf.math.conj(a_perp_r) +
        a_par_l * tf.math.conj(a_par_r)
    )


def _j1c(amplitudes, four_mass_mu_over_q2):
    [_, _, _, _, a_zero_l, a_zero_r] = amplitudes
    return tf.math.abs(a_zero_l) ** 2 + tf.math.abs(a_zero_r) ** 2 + \
        four_mass_mu_over_q2 * (2 * tf.math.real(a_zero_l * tf.math.conj(a_zero_r)))


def _j2s(amplitudes, beta2_mu):
    [a_par_l, a_par_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return (beta2_mu / four) * (
        tf.math.abs(a_perp_l) ** 2 + tf.math.abs(a_par_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 + tf.math.abs(a_par_r) ** 2
    )


def _j2c(amplitudes, beta2_mu):
    [_, _, _, _, a_zero_l, a_zero_r] = amplitudes
    return (- beta2_mu) * (tf.math.abs(a_zero_l) ** 2 + tf.math.abs(a_zero_r) ** 2)


def _j3(amplitudes, beta2_mu):
    [a_par_l, a_par_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return (beta2_mu / two) * (
        tf.math.abs(a_perp_l) ** 2 - tf.math.abs(a_par_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 - tf.math.abs(a_par_r) ** 2
    )


def _j4(amplitudes, beta2_mu):
    [a_par_l, a_par_r, _, _, a_zero_l, a_zero_r] = amplitudes
    return (beta2_mu / tf.sqrt(two)) * (
        tf.math.real(a_zero_l * tf.math.conj(a_par_l)) +
        tf.math.real(a_zero_r * tf.math.conj(a_par_r))
    )


def _j5(amplitudes, beta_mu):
    [_, _, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = amplitudes
    return tf.sqrt(two) * beta_mu * (
        tf.math.real(a_zero_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_zero_r * tf.math.conj(a_perp_r))
    )


def _j6s(amplitudes, beta_mu):
    [a_par_l, a_par_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return two * beta_mu * (
        tf.math.real(a_par_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_par_r * tf.math.conj(a_perp_r))
    )


def _j7(amplitudes, beta_mu):
    [a_par_l, a_par_r, _, _, a_zero_l, a_zero_r] = amplitudes
    return tf.sqrt(two) * beta_mu * (
        tf.math.imag(a_zero_l * tf.math.conj(a_par_l)) -
        tf.math.imag(a_zero_r * tf.math.conj(a_par_r))
    )


def _j8(amplitudes, beta2_mu):
    [_, _, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = amplitudes
    return (beta2_mu / tf.sqrt(two)) * (
        tf.math.imag(a_zero_l * tf.math.conj(a_perp_l)) +
        tf.math.imag(a_zero_r * tf.math.conj(a_perp_r))
    )


def _j9(amplitudes, beta2_mu):
    [a_par_l, a_par_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return beta2_mu * (
        tf.math.imag(tf.math.conj(a_par_l) * a_perp_l) +
        tf.math.imag(tf.math.conj(a_par_r) * a_perp_r)
    )


def _integrate_decay_rate(coeffs):
    return tf_integrate.odeint(
        lambda _, q2: _decay_rate_angle_integrated(q2, coeffs),
        0.0,
        tf.stack([q2_min, q2_max]),
        rtol=1e-4,
        atol=1e-2,
    )[1]


def _coeffs_to_amplitudes(q2, coeffs):
    amplitudes = []
    for amplitude_coeffs in coeffs:
        components = []
        for component_coeffs in amplitude_coeffs:
            components.append(component_coeffs[0] + (component_coeffs[1] * q2) + (component_coeffs[2] / q2))
        amplitudes.append(tf.complex(components[0], components[1]))

    return amplitudes
