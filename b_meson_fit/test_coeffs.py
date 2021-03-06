import itertools
import numpy.testing as nt
import tensorflow.compat.v2 as tf
import unittest

import b_meson_fit.coeffs as bmfc

tf.enable_v2_behavior()


class TestCoeffs(unittest.TestCase):

    # List of fit coefficient IDs we expect to be trainable
    fit_trainable_ids = list(itertools.chain(range(0, 21), range(24, 27), [36], [39], [42], [45]))

    def test_names(self):
        """Check coeff names"""
        self.assertEqual('a_para_l_re_alpha', bmfc.names[0])
        self.assertEqual(r'Re($A_{\parallel}^L$) $\alpha$', bmfc.latex_names[0])

        self.assertEqual('a_perp_r_re_beta', bmfc.names[19])
        self.assertEqual(r'Re($A_{\bot}^R$) $\beta$', bmfc.latex_names[19])

        self.assertEqual('a_0_r_im_gamma', bmfc.names[35])
        self.assertEqual(r'Im($A_{0}^R$) $\gamma$', bmfc.latex_names[35])

        self.assertEqual('a_00_l_im_gamma', bmfc.names[41])
        self.assertEqual(r'Im($A_{00}^L$) $\gamma$', bmfc.latex_names[41])

    def test_signal_coeffs(self):
        """Check signal coefficients have correct # of all constants"""
        signal = bmfc.signal(bmfc.SM)

        self.assertEqual(48, len(signal))
        for i in range(48):
            self.assertFalse(bmfc.is_trainable(signal[i]))

    def test_fit_coeffs_init_twice_largest_signal_same_sign(self):
        """Check fit coefficients have correct #, the right ones are trainable,
            and that randomisation/defaults work
        """
        largest_signal = [0.0] * bmfc.count
        for model in bmfc.signal_models:
            coeffs = bmfc.signal(model)
            for idx, coeff in enumerate(coeffs):
                if tf.math.abs(coeff).numpy() > tf.math.abs(largest_signal[idx]).numpy():
                    largest_signal[idx] = coeff

        fit = bmfc.fit(bmfc.FIT_INIT_TWICE_LARGEST_SIGNAL_SAME_SIGN)

        for i in range(48):
            if i in self.fit_trainable_ids:
                # Randomization should be enabled. Check from 0 to 2x largest signal
                if largest_signal[i].numpy() < 0:
                    random_min = 2.0 * largest_signal[i].numpy()
                    random_max = 0.0
                else:
                    random_min = 0.0
                    random_max = 2.0 * largest_signal[i].numpy()
                self.assertTrue(
                    random_min < fit[i].numpy() < random_max,
                    'Coeff {} fails {} < {} < {}'.format(i, random_min, fit[i].numpy(), random_max)
                )

    def test_fit_coeffs_init_twice_current_signal_any_sign(self):
        """Check fit coefficients have correct #, the right ones are trainable,
            and that randomisation/defaults work
        """
        signal = bmfc.signal(bmfc.SM)
        fit = bmfc.fit(bmfc.FIT_INIT_TWICE_CURRENT_SIGNAL_ANY_SIGN, bmfc.SM)

        for i in range(48):
            if i in self.fit_trainable_ids:
                # Randomization should be enabled. Check from 0 to 2x SM value
                random_min = -2.0 * tf.math.abs(signal[i]).numpy()
                random_max = 2.0 * tf.math.abs(signal[i]).numpy()
                self.assertTrue(
                    random_min < fit[i].numpy() < random_max,
                    'Coeff {} fails {} < {} < {}'.format(i, random_min, fit[i].numpy(), random_max)
                )

    def test_fit_coeffs_init_current_signal(self):
        """Check fit coefficients have correct #, the right ones are trainable,
            and that randomisation/defaults work
        """
        signal = bmfc.signal(bmfc.SM)
        fit = bmfc.fit(bmfc.FIT_INIT_CURRENT_SIGNAL, bmfc.SM)

        for i in range(48):
            if i in self.fit_trainable_ids:
                nt.assert_allclose(signal[i].numpy(), fit[i].numpy(), atol=0, rtol=1e-10)

    def test_fit_coeffs_init_constant(self):
        """Check fit coefficients have correct #, the right ones are trainable,
            and that randomisation/defaults work
        """
        fit = bmfc.fit(12.345)

        for i in range(48):
            if i in self.fit_trainable_ids:
                nt.assert_allclose(tf.constant(12.345).numpy(), fit[i].numpy(), atol=0, rtol=1e-10)

    def test_fit_coeffs_fix_p_wave(self):
        """Check fit coefficients works properly when passing `fix_p_wave_model`"""
        signal = bmfc.signal(bmfc.SM)
        fit = bmfc.fit(12.345, fix_p_wave_model=bmfc.SM)

        for i in range(48):
            if i in self.fit_trainable_ids:
                if i < 36:
                    # P-wave coeffs should be locked to the signal model values
                    nt.assert_allclose(signal[i].numpy(), fit[i].numpy(), atol=0, rtol=1e-10)
                    self.assertFalse(bmfc.is_trainable(fit[i]), 'Coeff {} should not be trainable'.format(i))
                else:
                    # S-wave coeffs should be set to the constant value and be trainable
                    nt.assert_allclose(tf.constant(12.345).numpy(), fit[i].numpy(), atol=0, rtol=1e-10)
                    self.assertTrue(bmfc.is_trainable(fit[i]), 'Coeff {} should be trainable'.format(i))

    def test_fit_coeffs_trainable(self):
        """Check fit coefficients have correct # and the right ones are trainable"""
        fit = bmfc.fit(12.345)

        self.assertEqual(48, len(fit))
        for i in range(48):
            if i in self.fit_trainable_ids:
                self.assertTrue(bmfc.is_trainable(fit[i]), 'Coeff {} should be trainable'.format(i))
            else:
                self.assertEqual(0.0, fit[i].numpy(), 'Coeff {} != 0.0'.format(i))
                self.assertFalse(bmfc.is_trainable(fit[i]), 'Coeff {} should not be trainable'.format(i))


if __name__ == '__main__':
    unittest.main()
