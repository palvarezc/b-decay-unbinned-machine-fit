#from scipy import integrate


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
