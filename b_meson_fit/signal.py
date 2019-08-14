"""Contains signal probability functions"""
import math
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import b_meson_fit.breit_wigner as bmfbw
import b_meson_fit.integrate as bmfi

tf.enable_v2_behavior()

q2_min = tf.constant(1.0)  # 1 (GeV/c^2)^2
q2_max = tf.constant(8.0)  # 8 (GeV/c^2)^2
mass_mu = tf.constant(0.1056583745)  # in 0.106 GeV/c^2

# Step size to use for integrating decay rate
integration_dt = tf.constant(0.0025)

# Integrated BW distributions to apply to amplitudes
bw_k700 = bmfbw.k700_distribution_integrated()
bw_k892 = bmfbw.k892_distribution_integrated()
bw_k700_k892 = bmfbw.k700_k892_distribution_integrated()


def nll(coeffs, events):
    """
    Return negative of the log likelihood for given events based on particular amplitude coefficients

    Args:
        coeffs: List of scalar coefficient tensors
        events: Tensor of shape (N, 4) with axis 1 representing params [q2, cos_theta_k, cos_theta_l, phi]

    Returns:
        Rank-1 tensor with shape (N)
    """
    return -tf.reduce_sum(
        tf.math.log(
            pdf(coeffs, events)
        )
    )


def normalized_nll(coeffs, events):
    """Get the negative log likelihood divided by the signal event count

    Args:
        coeffs: List of scalar coefficient tensors
        events: Tensor of shape (N, 4) with axis 1 representing params [q2, cos_theta_k, cos_theta_l, phi]

    Returns:
        Rank-1 tensor with shape (N)
    """
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
    events = []
    events_found = 0

    # Find the integrated decay rate so we can normalise our decay rates to probabilities
    norm = integrate_decay_rate(coeffs)

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
        probabilities = decay_rate(coeffs, candidates) / norm

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


def pdf(coeffs, events):
    """
    Return probabilities for given events based on particular amplitude coefficients

    Args:
        coeffs: List of scalar coefficient tensors
        events: Tensor of shape (N, 4) with axis 1 representing params [q2, cos_theta_k, cos_theta_l, phi]

    Returns:
        Rank-1 tensor with shape (N)
    """
    decay_rates = decay_rate(coeffs, events)
    norm = integrate_decay_rate(coeffs)
    # We don't want -ve probabilities, however 0 causes inf problems so floor at 1e-30
    return tf.math.maximum(decay_rates / norm, 1e-30)


def decay_rate(coeffs, events):
    """
    Calculate the decay rates for given events based on particular amplitude coefficients

    Args:
        coeffs: List of scalar coefficient tensors
        events: Tensor of shape (N, 4) with axis 1 representing params [q2, cos_theta_k, cos_theta_l, phi]

    Returns:
        Rank-1 tensor with shape (N)
    """
    [q2, cos_theta_k, cos_theta_l, phi] = tf.unstack(events, axis=1)
    amplitudes = coeffs_to_amplitudes(coeffs, q2)

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
    four_mass2_over_q2_ = four_mass2_over_q2(q2)
    beta2_mu_ = beta2_mu(four_mass2_over_q2_)
    beta_mu = tf.sqrt(beta2_mu_)

    # P-wave observables
    j1s_ = j1s(amplitudes, beta2_mu_, four_mass2_over_q2_)
    j1c_ = j1c(amplitudes, four_mass2_over_q2_)
    j2s_ = j2s(amplitudes, beta2_mu_)
    j2c_ = j2c(amplitudes, beta2_mu_)
    j3_ = j3(amplitudes, beta2_mu_)
    j4_ = j4(amplitudes, beta2_mu_)
    j5_ = j5(amplitudes, beta_mu)
    j6s_ = j6s(amplitudes, beta_mu)
    j7_ = j7(amplitudes, beta_mu)
    j8_ = j8(amplitudes, beta2_mu_)
    j9_ = j9(amplitudes, beta2_mu_)

    p_wave = (9 / (32 * math.pi)) * (
        (j1s_ * sin2_theta_k) +
        (j1c_ * cos2_theta_k) +
        (j2s_ * sin2_theta_k * cos_2theta_l) +
        (j2c_ * cos2_theta_k * cos_2theta_l) +
        (j3_ * sin2_theta_k * sin2_theta_l * cos_2phi) +
        (j4_ * sin_2theta_k * sin_2theta_l * cos_phi) +
        (j5_ * sin_2theta_k * sin_theta_l * cos_phi) +
        (j6s_ * sin2_theta_k * cos_theta_l) +
        (j7_ * sin_2theta_k * sin_theta_l * sin_phi) +
        (j8_ * sin_2theta_k * sin_2theta_l * sin_phi) +
        (j9_ * sin2_theta_k * sin2_theta_l * sin_2phi)
    )

    # S-wave observables
    j1c_prime_ = j1c_prime(amplitudes)
    j1c_dblprime_ = j1c_dblprime(amplitudes)
    j4_prime_ = j4_prime(amplitudes)
    j5_prime_ = j5_prime(amplitudes)
    j7_prime_ = j7_prime(amplitudes)
    j8_prime_ = j8_prime(amplitudes)

    s_wave = (9 / (32 * math.pi)) * (
        (j1c_prime_ * (1 - cos_2theta_l)) +
        (j1c_dblprime_ * cos_theta_k * (1 - cos_2theta_l)) +
        (j4_prime_ * sin_2theta_l * sin_theta_k * cos_phi) +
        (j5_prime_ * sin_theta_l * sin_theta_k * cos_phi) +
        (j7_prime_ * sin_theta_l * sin_theta_k * sin_phi) +
        (j8_prime_ * sin_2theta_l * sin_theta_k * sin_phi)
    )

    return p_wave + s_wave


def decay_rate_angle_integrated(coeffs, q2):
    """
    Calculate the angle-integrated decay rates for given q^2 values based on particular amplitude coefficients

    Args:
        coeffs: List of scalar coefficient tensors
        q2: Rank-1 tensor of shape (N) with q^2 values

    Returns:
        Rank-1 tensor with shape (N)
    """
    amplitudes = coeffs_to_amplitudes(coeffs, q2)

    return decay_rate_angle_integrated_p_wave(amplitudes, q2) + decay_rate_angle_integrated_s_wave(amplitudes)


def decay_rate_frac_s(coeffs, q2):
    """
    Calculate the S-wave contribution fraction via decay_rate..() functions

    Args:
        coeffs: List of scalar coefficient tensors
        q2: Rank-1 tensor of shape (N) with q^2 values

    Returns:
        Rank-1 tensor with shape (N)
    """
    amplitudes = coeffs_to_amplitudes(coeffs, q2)

    p_wave = decay_rate_angle_integrated_p_wave(amplitudes, q2)
    s_wave = decay_rate_angle_integrated_s_wave(amplitudes)

    return s_wave / (p_wave + s_wave)


def modulus_frac_s(coeffs, q2):
    """
    Calculate the S-wave contribution fraction via squaring the moduli

    Comes from eqn. 8 of arXiv:1504.00574v2

    Args:
        coeffs: List of scalar coefficient tensors
        q2: Rank-1 tensor of shape (N) with q^2 values

    Returns:
        Rank-1 tensor with shape (N)
    """
    [a_para_l, a_para_r, a_perp_l, a_perp_r, a_0_l, a_0_r, a_00_l, a_00_r] = coeffs_to_amplitudes(coeffs, q2)

    # From eqn. 8 of arXiv:1504.00574v2
    return (
             (tf.math.abs(a_00_l) ** 2) * bw_k700 +
             (tf.math.abs(a_00_r) ** 2) * bw_k700
     ) / (
             (tf.math.abs(a_00_l) ** 2) * bw_k700 +
             (tf.math.abs(a_0_l) ** 2) * bw_k892 +
             (tf.math.abs(a_para_l) ** 2) * bw_k892 +
             (tf.math.abs(a_perp_l) ** 2) * bw_k892 +
             (tf.math.abs(a_00_r) ** 2) * bw_k700 +
             (tf.math.abs(a_0_r) ** 2) * bw_k892 +
             (tf.math.abs(a_para_r) ** 2) * bw_k892 +
             (tf.math.abs(a_perp_r) ** 2) * bw_k892
     )


def decay_rate_angle_integrated_p_wave(amplitudes, q2):
    """
    Calculate the P-wave contribution to the angle-integrated decay rate for given q^2 values and amplitudes

    Formula can be found in the Mathematica file "decay_rate.nb" and the paper arXiv:1202.4266
    """
    # Mass terms
    four_mass2_over_q2_ = four_mass2_over_q2(q2)
    beta2_mu_ = beta2_mu(four_mass2_over_q2_)

    # Observables
    j1s_ = j1s(amplitudes, beta2_mu_, four_mass2_over_q2_)
    j1c_ = j1c(amplitudes, four_mass2_over_q2_)
    j2s_ = j2s(amplitudes, beta2_mu_)
    j2c_ = j2c(amplitudes, beta2_mu_)

    return (1 / 4) * ((6 * j1s_) + (3 * j1c_) - (2 * j2s_) - j2c_)


def decay_rate_angle_integrated_s_wave(amplitudes):
    """
    Calculate the S-wave contribution to the angle-integrated decay rate for given q^2 values and amplitudes

    Formula can be found in the Mathematica file "decay_rate.nb"
    """
    # Observables
    j1c_prime_ = j1c_prime(amplitudes)

    return 3 * j1c_prime_


def four_mass2_over_q2(q2):
    """Calculate 4*m_μ^2/q^2"""
    return (4.0 * (mass_mu ** 2)) / q2


def beta2_mu(four_mass2_over_q2_):
    """Calculate β_μ^2"""
    return 1.0 - four_mass2_over_q2_


def j1s(amplitudes, beta2_mu_, four_mass2_over_q2_):
    """Calculate j1s angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _, _, _] = amplitudes
    return (
        ((2.0 + beta2_mu_) / 4.0) * (
            (tf.math.abs(a_perp_l) ** 2) +
            (tf.math.abs(a_para_l) ** 2) +
            (tf.math.abs(a_perp_r) ** 2) +
            (tf.math.abs(a_para_r) ** 2)
        ) + four_mass2_over_q2_ * tf.math.real(
            (a_perp_l * tf.math.conj(a_perp_r)) +
            (a_para_l * tf.math.conj(a_para_r))
        )
    ) * bw_k892


def j1c(amplitudes, four_mass2_over_q2_):
    """Calculate j1c angular observable"""
    [_, _, _, _, a_0_l, a_0_r, _, _] = amplitudes
    return (
        (tf.math.abs(a_0_l) ** 2) +
        (tf.math.abs(a_0_r) ** 2) +
        (four_mass2_over_q2_ * 2 * tf.math.real(a_0_l * tf.math.conj(a_0_r)))
    ) * bw_k892


def j2s(amplitudes, beta2_mu_):
    """Calculate j2s angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _, _, _] = amplitudes
    return (beta2_mu_ / 4.0) * (
        (tf.math.abs(a_perp_l) ** 2) +
        (tf.math.abs(a_para_l) ** 2) +
        (tf.math.abs(a_perp_r) ** 2) +
        (tf.math.abs(a_para_r) ** 2)
    ) * bw_k892


def j2c(amplitudes, beta2_mu_):
    """Calculate j2c angular observable"""
    [_, _, _, _, a_0_l, a_0_r, _, _] = amplitudes
    return (- beta2_mu_) * (
        (tf.math.abs(a_0_l) ** 2) +
        (tf.math.abs(a_0_r) ** 2)
    ) * bw_k892


def j3(amplitudes, beta2_mu_):
    """Calculate j3 angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _, _, _] = amplitudes
    return (beta2_mu_ / 2.0) * (
        (tf.math.abs(a_perp_l) ** 2) -
        (tf.math.abs(a_para_l) ** 2) +
        (tf.math.abs(a_perp_r) ** 2) -
        (tf.math.abs(a_para_r) ** 2)
    ) * bw_k892


def j4(amplitudes, beta2_mu_):
    """Calculate j4 angular observable"""
    [a_para_l, a_para_r, _, _, a_0_l, a_0_r, _, _] = amplitudes
    return (beta2_mu_ / tf.sqrt(2.0)) * (
        tf.math.real(a_0_l * tf.math.conj(a_para_l)) +
        tf.math.real(a_0_r * tf.math.conj(a_para_r))
    ) * bw_k892


def j5(amplitudes, beta_mu):
    """Calculate j5 angular observable"""
    [_, _, a_perp_l, a_perp_r, a_0_l, a_0_r, _, _] = amplitudes
    return tf.sqrt(2.0) * beta_mu * (
        tf.math.real(a_0_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_0_r * tf.math.conj(a_perp_r))
    ) * bw_k892


def j6s(amplitudes, beta_mu):
    """Calculate j6s angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _, _, _] = amplitudes
    return 2.0 * beta_mu * (
        tf.math.real(a_para_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_para_r * tf.math.conj(a_perp_r))
    ) * bw_k892


def j7(amplitudes, beta_mu):
    """Calculate j7 angular observable"""
    [a_para_l, a_para_r, _, _, a_0_l, a_0_r, _, _] = amplitudes
    return tf.sqrt(2.0) * beta_mu * (
        tf.math.imag(a_0_l * tf.math.conj(a_para_l)) -
        tf.math.imag(a_0_r * tf.math.conj(a_para_r))
    ) * bw_k892


def j8(amplitudes, beta2_mu_):
    """Calculate j8 angular observable"""
    [_, _, a_perp_l, a_perp_r, a_0_l, a_0_r, _, _] = amplitudes
    return (beta2_mu_ / tf.sqrt(2.0)) * (
        tf.math.imag(a_0_l * tf.math.conj(a_perp_l)) +
        tf.math.imag(a_0_r * tf.math.conj(a_perp_r))
    ) * bw_k892


def j9(amplitudes, beta2_mu_):
    """Calculate j9 angular observable"""
    [a_para_l, a_para_r, a_perp_l, a_perp_r, _, _, _, _] = amplitudes
    return beta2_mu_ * (
        tf.math.imag(tf.math.conj(a_para_l) * a_perp_l) +
        tf.math.imag(tf.math.conj(a_para_r) * a_perp_r)
    ) * bw_k892


def j1c_prime(amplitudes):
    """Calculate j'1c angular observable"""
    [_, _, _, _, _, _, a_00_l, a_00_r] = amplitudes
    return (1 / 3) * (
        (tf.math.abs(a_00_l) ** 2) +
        (tf.math.abs(a_00_r) ** 2)
    ) * bw_k700


def j1c_dblprime(amplitudes):
    """Calculate j''1c angular observable"""
    [_, _, _, _, a_0_l, a_0_r, a_00_l, a_00_r] = amplitudes
    return (2 / tf.sqrt(3.0)) * (
        tf.math.real(a_00_l * tf.math.conj(a_0_l) * bw_k700_k892) +
        tf.math.real(a_00_r * tf.math.conj(a_0_r) * bw_k700_k892)
    )


def j4_prime(amplitudes):
    """Calculate j'4 angular observable"""
    [a_para_l, a_para_r, _, _, _, _, a_00_l, a_00_r] = amplitudes
    return tf.sqrt(2.0 / 3.0) * (
        tf.math.real(a_00_l * tf.math.conj(a_para_l) * bw_k700_k892) +
        tf.math.real(a_00_r * tf.math.conj(a_para_r) * bw_k700_k892)
    )


def j5_prime(amplitudes):
    """Calculate j'5 angular observable"""
    [_, _, a_perp_l, a_perp_r, _, _, a_00_l, a_00_r] = amplitudes
    return 2 * tf.sqrt(2.0 / 3.0) * (
        tf.math.real(a_00_l * tf.math.conj(a_perp_l) * bw_k700_k892) -
        tf.math.real(a_00_r * tf.math.conj(a_perp_r) * bw_k700_k892)
    )


def j7_prime(amplitudes):
    """Calculate j'7 angular observable"""
    [a_para_l, a_para_r, _, _, _, _, a_00_l, a_00_r] = amplitudes
    return 2 * tf.sqrt(2.0 / 3.0) * (
        tf.math.imag(a_00_l * tf.math.conj(a_para_l) * bw_k700_k892) -
        tf.math.imag(a_00_r * tf.math.conj(a_para_r) * bw_k700_k892)
    )


def j8_prime(amplitudes):
    """Calculate j'8 angular observable"""
    [_, _, a_perp_l, a_perp_r, _, _, a_00_l, a_00_r] = amplitudes
    return tf.sqrt(2.0 / 3.0) * (
        tf.math.imag(a_00_l * tf.math.conj(a_perp_l) * bw_k700_k892) +
        tf.math.imag(a_00_r * tf.math.conj(a_perp_r) * bw_k700_k892)
    )


def integrate_decay_rate(coeffs):
    """
    Integrate previously angle integrated decay rate function over q^2 for particular amplitude coefficients
    """
    return bmfi.trapezoid(
        lambda q2: decay_rate_angle_integrated(coeffs, q2),
        q2_min,
        q2_max,
        integration_dt
    )


def coeffs_to_amplitudes(coeffs, q2):
    """
    Arrange flat list of coefficients into list of amplitudes

    Each amplitude is a tf.complex(re, im) object

    Each re and im component is an anzatz of α + β*q^2 + γ/q^2
    """
    def _anzatz(c):
        return c[0] + (c[1] * q2) + (c[2] / q2)

    def _amplitude(c):
        return tf.complex(_anzatz(c[0:3]), _anzatz(c[3:6]))

    return [_amplitude(coeffs[i:i+6]) for i in range(0, len(coeffs), 6)]
