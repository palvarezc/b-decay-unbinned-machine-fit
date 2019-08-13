import numpy.testing as nt
import tensorflow.compat.v2 as tf
import unittest

import b_meson_fit.breit_wigner as bmfbw

tf.enable_v2_behavior()


class TestBreitWigner(unittest.TestCase):

    # Fields are: name, integrated distribution function, expected value
    test_distributions = [
        ('k700', bmfbw.k700_distribution_integrated, 0.2956765294075012+0j,),
        ('k892', bmfbw.k892_distribution_integrated, 0.8878989219665527+0j,),
        ('k700/k892', bmfbw.k700_k892_distribution_integrated, 0.6396150588989258+0.009679232724010944j),
    ]

    def test_distributions_integrated(self):
        """
        Check that integrated Breit-Wigner distributions return expected values

        Expected values found from running the code, so the point of this test is to check that nothing breaks
        during refactoring.
        """
        # Check for different lists of coefficients
        for name, distribution_integrated, expected in self.test_distributions:
            with self.subTest(name=name):
                actual = distribution_integrated()
                nt.assert_allclose(expected.real, tf.math.real(actual).numpy(), atol=0, rtol=1e-10)
                nt.assert_allclose(expected.imag, tf.math.imag(actual).numpy(), atol=0, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
