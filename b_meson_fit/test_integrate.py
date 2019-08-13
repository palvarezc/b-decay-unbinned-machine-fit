import math
import numpy.testing as nt
import tensorflow.compat.v2 as tf
import unittest

import b_meson_fit.integrate as bmfi

tf.enable_v2_behavior()


class TestIntegrate(unittest.TestCase):

    # Fields are: name, integral, start, stop, expected
    test_integrals = [
        ('x^2', lambda x: x ** 2, -5.0, 5.0, 83.3333,),
        ('x^3', lambda x: x ** 3, -5.0, 5.0, 0.0,),
        ('sine', lambda x: tf.math.sin(x), 0.0, math.pi / 2, 1.0),
    ]

    def test_trapezoid(self):
        """
        Check that trapezoid() returns expected values. Uses 10_000 bins for each.
        """
        # Check for different lists of coefficients
        for name, integral, start, stop, expected in self.test_integrals:
            with self.subTest(name=name):
                actual = bmfi.trapezoid(
                    integral,
                    start,
                    stop,
                    (stop - start) / 1e4,
                )
                # Check within tolerance as we're using bins
                nt.assert_allclose(expected, actual.numpy(), atol=1e-4, rtol=0.001)

    def test_trapezoid_supports_float64(self):
        """
        Check that trapezoid() supports integrals that return tf.float64
        """
        actual = bmfi.trapezoid(
            lambda x: tf.cast(x + 3.0, dtype=tf.float64),
            -5.0,
            5.0,
            0.1,
        )
        self.assertEqual(tf.float64, actual.dtype, 'Check returned dtype is float64')
        # Check within tolerance as we're using bins
        nt.assert_allclose(30.0, actual.numpy(), atol=1e-4, rtol=0.001)

    def test_trapezoid_supports_complex64(self):
        """
        Check that trapezoid() supports integrals that return tf.complex64
        """
        actual = bmfi.trapezoid(
            lambda x: tf.complex(x + 3.0, x + 4.0),
            -5.0,
            5.0,
            0.1,
        )
        self.assertEqual(tf.complex64, actual.dtype, 'Check returned dtype is complex64')
        # Check within tolerance as we're using bins
        nt.assert_allclose(30.0, tf.math.real(actual).numpy(), atol=1e-4, rtol=0.001)
        nt.assert_allclose(40.0, tf.math.imag(actual).numpy(), atol=1e-4, rtol=0.001)


if __name__ == '__main__':
    unittest.main()
