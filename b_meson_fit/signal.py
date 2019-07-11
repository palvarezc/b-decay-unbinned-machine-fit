"""Contains signal probability functions"""

import math
import tensorflow.compat.v2 as tf
# Import this separately as its old Tensorflow v1 code
from tensorflow.contrib import integrate as tf_integrate
import tensorflow_probability as tfp

tf.enable_v2_behavior()

q2_min = tf.constant(1.0)  # 1 (GeV/c^2)^2
q2_max = tf.constant(8.0)  # 8 (GeV/c^2)^2
mass_mu = tf.constant(0.1056583745)  # in 0.106 GeV/c^2


@tf.function
def nll(coeffs, events):
    """
    Return negative of the log likelihood for given events based on particular amplitude coefficients

    Args:
        coeffs: List of scalar coefficient tensors
        events: Tensor of shape (N, 4) with axis 1 representing params [q2, cos_theta_k, cos_theta_l, phi]

    Returns:
        Rank-1 tensor with shape (N)
    """
    with tf.device('/device:GPU:0'):
        return -tf.reduce_sum(
            tf.math.log(
                _pdf(coeffs, events)
            )
        )


def normalized_nll(coeffs, events):
    """Get the negative log likelihood divided by the signal event count

    Returns:
        Rank-1 tensor with shape (N)
    """
    with tf.device('/device:GPU:0'):
        return nll(coeffs, events) / tf.cast(tf.shape(events)[0], tf.float32)


def generate(coeffs, events_total=100_000, batch_size=10_000_000):
    """
    Generate sample events based on particular amplitude coefficients

    Uses a Monte-Carlo accept-reject method: https://planetmath.org/acceptancerejectionmethod

    Args:
        coeffs: List of scalar coefficient tensors
        events_total: Number of events to generate
        batch_size: Number of event candidates to generate for each Monte-Carlo round

    Returns:
        Tensor of shape (events_total, 4) with axis 1 representing params [q2, cos_theta_k, cos_theta_l, phi]
    """
    with tf.device('/device:GPU:0'):
        events = []
        events_found = 0

        # Find the integrated decay rate so we can normalise our decay rates to probabilities
        norm = _integrate_decay_rate(coeffs)

        # Distributions for our independent variables
        q2_dist = tfp.distributions.Uniform(low=q2_min, high=q2_max)
        cos_theta_k_dist = tfp.distributions.Uniform(low=-1.0, high=1.0)
        cos_theta_l_dist = tfp.distributions.Uniform(low=-1.0, high=1.0)
        phi_dist = tfp.distributions.Uniform(low=-math.pi, high=math.pi)

        # Uniform distribution between 0 and 1 for candidate selection
        uniform_dist = tfp.distributions.Uniform(low=0.0, high=1.0)

        while events_found < events_total:
            # Get batch_size number of candidate events in tensor of shape (batch_size, 4)
            candidates = tf.stack(
                [
                    q2_dist.sample(batch_size),
                    cos_theta_k_dist.sample(batch_size),
                    cos_theta_l_dist.sample(batch_size),
                    phi_dist.sample(batch_size)
                ],
                axis=1
            )
            # Get a list of probabilities for each event with shape(batch_size,)
            probabilities = _decay_rate(coeffs, candidates) / norm

            # Get a uniform distribution of numbers between 0 and 1 with shape (batch_size,)
            uniforms = uniform_dist.sample(batch_size)

            # Get a list of row indexes for probabilities tensor (and therefore candidates tensor)
            #  where we accept them (uniform value < probability value)
            accept_candidates_ids = tf.squeeze(tf.where(tf.less(uniforms, probabilities)), -1)
            # Use indexes to gather candidates we accept
            accept_candidates = tf.gather(candidates, accept_candidates_ids)

            # Append accepted candidates to our events list
            events.append(accept_candidates)
            events_found = events_found + tf.shape(accept_candidates)[0]

        # Bolt our event tensors together and limit to returning events_total rows
        return tf.concat(events, axis=0)[0:events_total, :]


def _pdf(coeffs, events):
    """Return probabilities for given events based on particular amplitude coefficients"""
    decay_rates = _decay_rate(coeffs, events)
    norm = _integrate_decay_rate(coeffs)
    return tf.math.maximum(decay_rates / norm, 1e-30)


def _decay_rate(coeffs, events):
    """Calculate the decay rates for given events based on particular amplitude coefficients"""
    [q2, cos_theta_k, cos_theta_l, phi] = tf.unstack(events, axis=1)
    amplitudes = _coeffs_to_amplitudes(coeffs, q2)

    # Angles
    cos2_theta_k = cos_theta_k ** 2
    sin2_theta_k = 1.0 - cos2_theta_k
    sin_theta_k = tf.sqrt(sin2_theta_k)
    sin_2theta_k = 2.0 * sin_theta_k * cos_theta_k

    cos2_theta_l = cos_theta_l ** 2
    cos_2theta_l = (2.0 * cos2_theta_l) - 1.0
    sin2_theta_l = 1.0 - cos2_theta_l
    sin_theta_l = tf.sqrt(sin2_theta_l)
    sin_2theta_l = 2.0 * sin_theta_l * cos_theta_l

    cos_phi = tf.math.cos(phi)
    cos_2phi = tf.math.cos(2.0 * phi)
    sin_phi = tf.math.sin(phi)
    sin_2phi = tf.math.sin(2.0 * phi)

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
        (j9 * sin2_theta_k * sin2_theta_l * sin_2phi)
    )


def _decay_rate_angle_integrated(coeffs, q2):
    """
    Calculate the angle-integrated decay rates for given q^2 values based on particular amplitude coefficients

    Formula can be found in the Mathematica file "n_sig.nb" and the paper arXiv:1202.4266
    """
    amplitudes = _coeffs_to_amplitudes(coeffs, q2)

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
    """Calculate 4*m_μ^2/q^2"""
    return (4.0 * (mass_mu ** 2)) / q2


def _beta2(four_mass2_over_q2):
    """Calculate β_μ^2"""
    return 1.0 - four_mass2_over_q2


def _j1s(amplitudes, beta2_mu, four_mass2_over_q2):
    """Calculate j1s angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return ((2.0 + beta2_mu) / 4.0) * (
        tf.math.abs(a_perp_l) ** 2 + tf.math.abs(a_para_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 + tf.math.abs(a_para_r) ** 2
    ) + four_mass2_over_q2 * tf.math.real(
        a_perp_l * tf.math.conj(a_perp_r) +
        a_para_l * tf.math.conj(a_para_r)
    )


def _j1c(amplitudes, four_mass2_over_q2):
    """Calculate j1c angular observable"""
    [_, _, _, _, a_zero_l, a_zero_r] = amplitudes
    return tf.math.abs(a_zero_l) ** 2 + tf.math.abs(a_zero_r) ** 2 + \
        four_mass2_over_q2 * (2 * tf.math.real(a_zero_l * tf.math.conj(a_zero_r)))


def _j2s(amplitudes, beta2_mu):
    """Calculate j2s angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return (beta2_mu / 4.0) * (
        tf.math.abs(a_perp_l) ** 2 + tf.math.abs(a_para_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 + tf.math.abs(a_para_r) ** 2
    )


def _j2c(amplitudes, beta2_mu):
    """Calculate j2c angular observable"""
    [_, _, _, _, a_zero_l, a_zero_r] = amplitudes
    return (- beta2_mu) * (tf.math.abs(a_zero_l) ** 2 + tf.math.abs(a_zero_r) ** 2)


def _j3(amplitudes, beta2_mu):
    """Calculate j3 angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return (beta2_mu / 2.0) * (
        tf.math.abs(a_perp_l) ** 2 - tf.math.abs(a_para_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 - tf.math.abs(a_para_r) ** 2
    )


def _j4(amplitudes, beta2_mu):
    """Calculate j4 angular observable"""
    [a_para_l, a_para_r, _, _, a_zero_l, a_zero_r] = amplitudes
    return (beta2_mu / tf.sqrt(2.0)) * (
        tf.math.real(a_zero_l * tf.math.conj(a_para_l)) +
        tf.math.real(a_zero_r * tf.math.conj(a_para_r))
    )


def _j5(amplitudes, beta_mu):
    """Calculate j5 angular observable"""
    [_, _, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = amplitudes
    return tf.sqrt(2.0) * beta_mu * (
        tf.math.real(a_zero_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_zero_r * tf.math.conj(a_perp_r))
    )


def _j6s(amplitudes, beta_mu):
    """Calculate j6s angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return 2.0 * beta_mu * (
        tf.math.real(a_para_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_para_r * tf.math.conj(a_perp_r))
    )


def _j7(amplitudes, beta_mu):
    """Calculate j7 angular observable"""
    [a_para_l, a_para_r, _, _, a_zero_l, a_zero_r] = amplitudes
    return tf.sqrt(2.0) * beta_mu * (
        tf.math.imag(a_zero_l * tf.math.conj(a_para_l)) -
        tf.math.imag(a_zero_r * tf.math.conj(a_para_r))
    )


def _j8(amplitudes, beta2_mu):
    """Calculate j8 angular observable"""
    [_, _, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = amplitudes
    return (beta2_mu / tf.sqrt(2.0)) * (
        tf.math.imag(a_zero_l * tf.math.conj(a_perp_l)) +
        tf.math.imag(a_zero_r * tf.math.conj(a_perp_r))
    )


def _j9(amplitudes, beta2_mu):
    """Calculate j9 angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _] = amplitudes
    return beta2_mu * (
        tf.math.imag(tf.math.conj(a_para_l) * a_perp_l) +
        tf.math.imag(tf.math.conj(a_para_r) * a_perp_r)
    )


def _integrate_decay_rate(coeffs):
    """Integrate decay rate function over all angles and a q^2 range for particular amplitude coefficients"""
    return tf_integrate.odeint(
        lambda _, q2: _decay_rate_angle_integrated(coeffs, q2),
        0.0,
        tf.stack([q2_min, q2_max]),
        # Tolerances chosen to maximum speed without changing accuracy much (~ < 0.07% drop depending on coeffs)
        rtol=1e-2,
        atol=1e0,
    )[1]


def _coeffs_to_amplitudes(coeffs, q2):
    """
    Arrange flat list of coefficients into list of amplitudes

    Each amplitude is a tf.complex(re, im) object

    Each re and im param is an anzatz of α + β*q^2 + γ/q^2
    """
    def _anzatz(c):
        return c[0] + (c[1] * q2) + (c[2] / q2)

    def _amplitude(c):
        return tf.complex(_anzatz(c[0:3]), _anzatz(c[3:6]))

    return [_amplitude(coeffs[i:i+6]) for i in range(0, len(coeffs), 6)]
