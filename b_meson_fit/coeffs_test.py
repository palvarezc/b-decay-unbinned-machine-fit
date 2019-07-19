import itertools
import unittest

import b_meson_fit.coeffs as bmfc


class TestCoeffs(unittest.TestCase):

    # List of fit coefficient IDs we expect to be trainable
    fit_trainable_ids = list(itertools.chain(range(0, 21), range(24, 27), [36], [39], [42], [45]))
    # Name/default for checking coefficient defaults works
    fit_coeff_defaults = [
        ('ones', 1.0),
        ('random', None)
    ]

    def test_names(self):
        """Check coeff names"""
        self.assertEqual('a_para_l_re_alpha', bmfc.names[0])
        self.assertEqual(r'Re($a_{\parallel}^L$) $\alpha$', bmfc.latex_names[0])

        self.assertEqual('a_perp_r_re_beta', bmfc.names[19])
        self.assertEqual(r'Re($a_{\bot}^R$) $\beta$', bmfc.latex_names[19])

        self.assertEqual('a_0_r_im_gamma', bmfc.names[35])
        self.assertEqual(r'Im($a_{0}^R$) $\gamma$', bmfc.latex_names[35])

        self.assertEqual('a_00_l_im_gamma', bmfc.names[41])
        self.assertEqual(r'Im($a_{00}^L$) $\gamma$', bmfc.latex_names[41])

    def test_signal_coeffs(self):
        """Check signal coefficients have correct # of all constants"""
        signal = bmfc.signal()

        self.assertEqual(48, len(signal))
        for i in range(48):
            self.assertFalse(bmfc.is_trainable(signal[i]))

    def test_fit_coeffs(self):
        """Check fit coefficients have correct #, the right ones are trainable,
            and that randomisation/defaults work
        """
        signal = bmfc.signal()

        for name, default in self.fit_coeff_defaults:
            with self.subTest(name=name):
                bmfc.fit_default = default

                fit = bmfc.fit()

                self.assertEqual(48, len(fit))
                for i in range(48):
                    if i in self.fit_trainable_ids:
                        if default is not None:
                            # Randomization should be disabled
                            self.assertEqual(default, fit[i].numpy(), 'Coeff {} != {}'.format(i, default))
                        else:
                            # Randomization should be enabled. Check within +/- 100% of signal
                            if signal[i].numpy() < 0:
                                random_min = 2.0 * signal[i].numpy()
                                random_max = 0.0
                            else:
                                random_min = 0.0
                                random_max = 2.0 * signal[i].numpy()
                            self.assertTrue(
                                random_min < fit[i].numpy() < random_max,
                                'Coeff {} fails {} < {} < {}'.format(i, random_min, fit[i].numpy(), random_max)
                            )
                        self.assertTrue(bmfc.is_trainable(fit[i]), 'Coeff {} should be trainable'.format(i))
                    else:
                        self.assertEqual(0.0, fit[i].numpy(), 'Coeff {} != 0.0'.format(i))
                        self.assertFalse(bmfc.is_trainable(fit[i]), 'Coeff {} should not be trainable'.format(i))