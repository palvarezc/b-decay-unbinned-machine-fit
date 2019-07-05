

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
