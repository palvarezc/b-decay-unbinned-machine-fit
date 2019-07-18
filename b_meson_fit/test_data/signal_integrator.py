#!/usr/bin/env python
"""
Helper script to generate test decay rate values for signal_test.py.

Go through our test_coeffs data provider and use odeint_fixed () to across all variables to work out a
decay rate.

Ideally this would use odeint(), in which case the test_decay_rate_integration_methods_approx_equal tolerance
could be reduced. However odeint() results in a 'underflow in dt' error with this data so the value is
approximated through odeint_fixed().

Will take a long while to run.
"""
import math
import tensorflow.compat.v2 as tf
# Import this separately as its old Tensorflow v1 code
from tensorflow.contrib import integrate as tf_integrate

import b_meson_fit.script as bmfr
import b_meson_fit.signal as bmfs
import b_meson_fit.signal_test as bmfst

tf.enable_v2_behavior()

# Ranges we want to integrate over
q2_range = tf.cast([bmfs.q2_min, bmfs.q2_max], dtype=tf.float32)
cos_k_range = tf.constant([-1.0, 1.0], dtype=tf.float32)
cos_l_range = tf.constant([-1.0, 1.0], dtype=tf.float32)
phi_range = tf.constant([-math.pi, math.pi], dtype=tf.float32)

# When integrating approximate each variable into bins. Bin sizes found through trial and error
dt = lambda r, bins: (r[1] - r[0]) / bins
q2_dt = dt(q2_range, 20)
cos_k_dt = dt(cos_k_range, 15)
cos_l_dt = dt(cos_l_range, 20)
phi_dt = dt(phi_range, 6)

# Massively improve the speed of this script by autographing our decay_rate() function.
decay_rate = tf.function(bmfs.decay_rate)

with bmfr.Script() as script:
    # Check for different lists of coefficients
    for c_name, coeffs, expected_decay_rate in bmfst.TestSignal.test_coeffs:
        bmfr.stdout('Integrating {}'.format(c_name))

        # Integrate decay_rate() over the 4 independent variables
        # odeint_fixed() is used as the faster and more accurate odeint() resulted either NaNs or float underflows
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

        found = full_integrated_rate
        ratio = tf.math.maximum(found, expected_decay_rate) / tf.math.minimum(found, expected_decay_rate)

        bmfr.stdout(
            'Integrated {}. Found: {} Expected: {} Ratio: {}'.format(
                c_name,
                full_integrated_rate.numpy(),
                expected_decay_rate,
                ratio
            )
        )
