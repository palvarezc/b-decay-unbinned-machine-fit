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


@tf.function
def pdf(events, coeffs):
    with tf.device('/device:GPU:0'):
        decay_rates = _decay_rate(events, coeffs)
        norm = _integrate_decay_rate(coeffs)
        return tf.math.maximum(decay_rates / norm, 1e-30)


@tf.function
def negative_log_likelihood(events, coeffs):
    with tf.device('/device:GPU:0'):
        return -tf.reduce_sum(
            tf.math.log(
                pdf(events, coeffs)
            )
        )


def generate_events(events_num, coeffs):
    with tf.device('/device:GPU:0'):
        q2 = tfp.distributions.Uniform(low=q2_min, high=q2_max).sample(options_num)
        cos_theta_k = tfp.distributions.Uniform(low=-1.0, high=1.0).sample(options_num)
        cos_theta_l = tfp.distributions.Uniform(low=-1.0, high=1.0).sample(options_num)
        phi = tfp.distributions.Uniform(low=-math.pi, high=math.pi).sample(options_num)

        options = tf.stack([q2, cos_theta_k, cos_theta_l, phi], axis=1)

        probs = pdf(options, coeffs)
        normalised_probs = pdf(options, coeffs) / tf.reduce_sum(probs)
        choices = np.random.choice(options.get_shape()[0], events_num, p=normalised_probs.numpy())

        return tf.gather(options, choices, name='events')


def _decay_rate(events, coeffs):
    [q2, cos_theta_k, cos_theta_l, phi] = tf.unstack(events, axis=1)
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
    four_mass2_over_q2 = _four_mass2_over_q2(q2)
    beta2 = _beta2(four_mass2_over_q2)
    beta = tf.sqrt(beta2)

    # Observables
    j1s = _j1s(amplitudes, beta2, four_mass2_over_q2)
    j1c = _j1c(amplitudes, four_mass2_over_q2)
    j2s = _j2s(amplitudes, beta2)
    j2c = _j2c(amplitudes, beta2)
    j3 = _j3(amplitudes, beta2)
    j4 = _j4(amplitudes, beta2)
    j5 = _j5(amplitudes, beta)
    j6s = _j6s(amplitudes, beta)
    j7 = _j7(amplitudes, beta)
    j8 = _j8(amplitudes, beta2)
    j9 = _j9(amplitudes, beta2)

    return (9 / (32 * math.pi)) * (
        (j1s * sin2_theta_k) +
        (j1c * cos2_theta_k) +
        (j2s * sin2_theta_k * cos_2theta_l) +
        (j2c * cos2_theta_k * cos_2theta_l) +
        (j3 * sin2_theta_k * sin2_theta_l * cos_2phi) +
        (j4 * sin_2theta_k * sin_2theta_l * cos_phi) +
        (j5 * sin_2theta_k * sin_theta_l * cos_phi) +
        (j6s * sin2_theta_k * cos_theta_l) +
        (j7 * sin_2theta_k * sin_theta_l * sin_phi) +
        (j8 * sin_2theta_k * sin_2theta_l * sin_phi) +
        (j9 * sin_2theta_k * sin_2theta_l * sin_2phi)
    )


# https://arxiv.org/abs/1202.4266
# @see notebook
def _decay_rate_angle_integrated(q2, coeffs):
    amplitudes = _coeffs_to_amplitudes(q2, coeffs)

    # Mass terms
    four_mass2_over_q2 = _four_mass2_over_q2(q2)
    beta2 = _beta2(four_mass2_over_q2)

    # Observables
    j1s = _j1s(amplitudes, beta2, four_mass2_over_q2)
    j1c = _j1c(amplitudes, four_mass2_over_q2)
    j2s = _j2s(amplitudes, beta2)
    j2c = _j2c(amplitudes, beta2)

    return (1 / 4) * (
        (6 * j1s) +
        (3 * j1c) -
        (2 * j2s) -
        j2c
    )


def _four_mass2_over_q2(q2):
    return (four * (mass_mu ** 2)) / q2


def _beta2(four_mass2_over_q2):
    return one - four_mass2_over_q2


def _j1s(amplitudes, beta2_mu, four_mass2_over_q2):
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return ((two + beta2_mu) / four) * (
        tf.math.abs(a_perp_l) ** 2 + tf.math.abs(a_para_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 + tf.math.abs(a_para_r) ** 2
    ) + four_mass2_over_q2 * tf.math.real(
        a_perp_l * tf.math.conj(a_perp_r) +
        a_para_l * tf.math.conj(a_para_r)
    )


def _j1c(amplitudes, four_mass2_over_q2):
    [_, _, _, _, a_zero_l, a_zero_r] = amplitudes
    return tf.math.abs(a_zero_l) ** 2 + tf.math.abs(a_zero_r) ** 2 + \
        four_mass2_over_q2 * (2 * tf.math.real(a_zero_l * tf.math.conj(a_zero_r)))


def _j2s(amplitudes, beta2_mu):
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return (beta2_mu / four) * (
        tf.math.abs(a_perp_l) ** 2 + tf.math.abs(a_para_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 + tf.math.abs(a_para_r) ** 2
    )


def _j2c(amplitudes, beta2_mu):
    [_, _, _, _, a_zero_l, a_zero_r] = amplitudes
    return (- beta2_mu) * (tf.math.abs(a_zero_l) ** 2 + tf.math.abs(a_zero_r) ** 2)


def _j3(amplitudes, beta2_mu):
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return (beta2_mu / two) * (
        tf.math.abs(a_perp_l) ** 2 - tf.math.abs(a_para_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 - tf.math.abs(a_para_r) ** 2
    )


def _j4(amplitudes, beta2_mu):
    [a_para_l, a_para_r, _, _, a_zero_l, a_zero_r] = amplitudes
    return (beta2_mu / tf.sqrt(two)) * (
        tf.math.real(a_zero_l * tf.math.conj(a_para_l)) +
        tf.math.real(a_zero_r * tf.math.conj(a_para_r))
    )


def _j5(amplitudes, beta_mu):
    [_, _, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = amplitudes
    return tf.sqrt(two) * beta_mu * (
        tf.math.real(a_zero_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_zero_r * tf.math.conj(a_perp_r))
    )


def _j6s(amplitudes, beta_mu):
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return two * beta_mu * (
        tf.math.real(a_para_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_para_r * tf.math.conj(a_perp_r))
    )


def _j7(amplitudes, beta_mu):
    [a_para_l, a_para_r, _, _, a_zero_l, a_zero_r] = amplitudes
    return tf.sqrt(two) * beta_mu * (
        tf.math.imag(a_zero_l * tf.math.conj(a_para_l)) -
        tf.math.imag(a_zero_r * tf.math.conj(a_para_r))
    )


def _j8(amplitudes, beta2_mu):
    [_, _, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = amplitudes
    return (beta2_mu / tf.sqrt(two)) * (
        tf.math.imag(a_zero_l * tf.math.conj(a_perp_l)) +
        tf.math.imag(a_zero_r * tf.math.conj(a_perp_r))
    )


def _j9(amplitudes, beta2_mu):
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return beta2_mu * (
        tf.math.imag(tf.math.conj(a_para_l) * a_perp_l) +
        tf.math.imag(tf.math.conj(a_para_r) * a_perp_r)
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
    def _anzatz(c):
        return c[0] + (c[1] * q2) + (c[2] / q2)

    def _amplitude(c):
        return tf.complex(_anzatz(c[0:3]), _anzatz(c[3:6]))

    return [_amplitude(coeffs[i:i+6]) for i in range(0, len(coeffs), 6)]
