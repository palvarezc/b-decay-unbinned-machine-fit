import math
import numpy.testing as nt
import tensorflow.compat.v2 as tf
# Import this separately as its old Tensorflow v1 code
from tensorflow.contrib import integrate as tf_integrate
import unittest

import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs

tf.enable_v2_behavior()


class TestSignal(unittest.TestCase):

    test_coeffs = [
        ('signal', bmfc.signal(),),
        ('ones', [tf.constant(1.0)] * 36,),
        ('integers', [tf.constant(float(i)) for i in range(-18, 18)],),
    ]

    def test_decay_rate_integration_methods_approx_equal(self):
        """
        Check that integrating the decay_rate() function across all variables is approximately equal to
        the _integrate_decay_rate() method that integrates a previously partially analytically integrated
        expression only over q^2

        Note this test takes ~4m40s to run (YMMV on other hardware)
        """
        # Ranges we want to integrate over
        q2_range = tf.cast([bmfs.q2_min, bmfs.q2_max], dtype=tf.float32)
        cos_k_range = tf.constant([-1.0, 1.0], dtype=tf.float32)
        cos_l_range = tf.constant([-1.0, 1.0], dtype=tf.float32)
        phi_range = tf.constant([-math.pi, math.pi], dtype=tf.float32)

        # When integrating approximate each variable into bins. Bin sizes found through trial and error
        def dt(r, bins): return (r[1] - r[0]) / bins
        q2_dt = dt(q2_range, 10)
        cos_k_dt = dt(cos_k_range, 8)
        cos_l_dt = dt(cos_l_range, 6)
        phi_dt = dt(phi_range, 5)

        # Massively improve the speed of the test by autographing our decay_rate() function. This does
        #  unfortunately make the test harder to debug
        decay_rate = tf.function(bmfs._decay_rate)

        # Check for different lists of coefficients
        for c_name, coeffs in self.test_coeffs:
            with self.subTest(c_name=c_name):

                with tf.device('/device:GPU:0'):
                    # Integrate decay_rate() over the 4 independent variables
                    # odeint_fixed() is used as odeint() resulted in a float underflow
                    full_integrated_rate = tf_integrate.odeint_fixed(
                        lambda _, q2: tf_integrate.odeint_fixed(
                            lambda _, cos_theta_k: tf_integrate.odeint_fixed(
                                lambda _, cos_theta_l: tf_integrate.odeint_fixed(
                                    lambda _, phi: decay_rate(
                                        coeffs,
                                        tf.expand_dims(tf.stack([q2, cos_theta_k, cos_theta_l, phi]), 0)
                                    )[0],
                                    0.0,
                                    phi_range,
                                    phi_dt,
                                    method='midpoint'
                                )[1],
                                0.0,
                                cos_l_range,
                                cos_l_dt,
                                method='midpoint'
                            )[1],
                            0.0,
                            cos_k_range,
                            cos_k_dt,
                            method='midpoint'

                        )[1],
                        0.0,
                        q2_range,
                        q2_dt,
                        method='midpoint'
                    )[1]

                    # Integrate partially analytically integrated expression over q^2
                    q2_only_integrated_rate = bmfs._integrate_decay_rate(coeffs)

                # Check values are the same to within 2%
                nt.assert_allclose(full_integrated_rate.numpy(), q2_only_integrated_rate.numpy(), atol=0, rtol=0.02)

    def test_integral_decay_rate_within_tolerance(self):
        """
        Check that the tolerances set in _integrate_decay_rate() have not been relaxed so much that
        they mess up the accuracy more than 0.07% from the true value.
        """
        for c_name, coeffs in self.test_coeffs:
            with self.subTest(c_name=c_name):
                true = tf_integrate.odeint(
                    lambda _, q2: bmfs._decay_rate_angle_integrated(coeffs, q2),
                    0.0,
                    tf.stack([bmfs.q2_min, bmfs.q2_max]),
                )[1]

                ours = bmfs._integrate_decay_rate(coeffs)

                nt.assert_allclose(true.numpy(), ours.numpy(), atol=0, rtol=0.0007)

    def test_generate_returns_correct_shape(self):
        """Check generate() returns a tensor of shape (events_total, 4)"""
        events = bmfs.generate(bmfc.signal(), 123_456)
        self.longMessage = True
        self.assertEqual(123_456, tf.shape(events)[0].numpy())
        self.assertEqual(4, tf.shape(events)[1].numpy())


if __name__ == '__main__':
    unittest.main()
