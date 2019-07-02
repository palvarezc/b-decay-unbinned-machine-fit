#from scipy import integrate


def integrated_decay_rate(q2, amplitudes):
    # https://arxiv.org/abs/1202.4266
    # @see notebook

    [a_par_l, a_par_r, a_perp_l, a_perp_r, a_zero_l, a_zero_r] = amplitudes

    # Mass terms
    four_mass_mu_over_q2 = (four * (mass_mu ** 2)) / q2
    beta2_mu = one - four_mass_mu_over_q2

    # Angular observables
    j_1s = ((two + beta2_mu) / four) * (
        tf.math.abs(a_perp_l) ** 2 + tf.math.abs(a_par_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 + tf.math.abs(a_par_r) ** 2
    ) + four_mass_mu_over_q2 * tf.math.real(
        a_perp_l * tf.math.conj(a_perp_r) +
        a_par_l * tf.math.conj(a_par_r)
    )

    j_1c = tf.math.abs(a_zero_l) ** 2 + tf.math.abs(a_zero_r) ** 2 + \
        four_mass_mu_over_q2 * (2 * tf.math.real(a_zero_l * tf.math.conj(a_zero_r)))

    j_2s = (beta2_mu / four) * (
        tf.math.abs(a_perp_l) ** 2 + tf.math.abs(a_par_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 + tf.math.abs(a_par_r) ** 2
    )

    j_2c = (- beta2_mu) * (tf.math.abs(a_zero_l) ** 2 + tf.math.abs(a_zero_r) ** 2)

    observables = [
        j_1s,
        j_1c,
        j_2s,
        j_2c,
    ]

    rate = (1/4) * ((3 * j_1c) + (6 * j_1s) - j_2c - (2 * j_2s))

    return [observables, rate]

#
# def n_sig(n_avg_dat, amplitude_coeffs):
#
#     def _integrate(q2, coeffs):
#         amplitudes = coeffs_to_amplitudes(q2, coeffs)
#         [_, rate_function] = integrated_decay_rate(q2, amplitudes)
#         return rate_function
#
#     n_sig_val = (n_avg_dat / (q2_max - q2_min)) * \
#         integrate.quad(_integrate, q2_min, q2_max, args=amplitude_coeffs, full_output=True)[0]
#
#     print(n_sig_val)
#
#     return n_sig_val

#n = n_sig(100_000, signal_coeffs)
