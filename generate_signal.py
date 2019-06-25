import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import tensorflow_probability as tfp

from tensorflow.python import tf2
if not tf2.enabled():
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions

# TODO: Check signal terms
# TODO: Basis fixing
# TODO: Convert to tf distribution?
# TODO: Pip requirements.txt (& test)
# TODO: Do fitting
# TODO: Split files/Unit tests/comments/doc comments

mass_mu = tf.constant(105.6583745e6)  # in 106 MeV/c^2
q2_min = tf.constant(2.0e18)  # 2 (GeV/c^2)^2
q2_max = tf.constant(6.0e18)  # 6 (GeV/c^2)^2


def decay_rate(independent_vars):
    q2 = independent_vars[:, 0]
    cos_theta_k = independent_vars[:, 1]
    cos_theta_l = independent_vars[:, 2]
    phi = independent_vars[:, 3]

    one = tf.constant(1.0)
    two = tf.constant(2.0)
    four = tf.constant(4.0)

    cos2_theta_k = cos_theta_k**2
    sin2_theta_k = one - cos2_theta_k
    sin_theta_k = tf.sqrt(sin2_theta_k)
    sin_2theta_k = two * sin_theta_k * cos_theta_k

    cos2_theta_l = cos_theta_l**2
    cos_2theta_l = (two * cos2_theta_l) - one
    sin2_theta_l = one - cos2_theta_l
    sin_theta_l = tf.sqrt(sin2_theta_l)
    sin_2theta_l = two * sin_theta_l * cos_theta_l

    cos_phi = tf.math.cos(phi)
    cos_2phi = tf.math.cos(two * phi)
    sin_phi = tf.math.sin(phi)
    sin_2phi = tf.math.sin(two * phi)

    four_mass_mu_over_q2 = (four * (mass_mu**2)) / q2
    beta2_mu = one - four_mass_mu_over_q2
    beta_mu = tf.sqrt(beta2_mu)

    a_par_l = tf.complex(one + (0 * q2) + (0 / q2), one + (0 * q2) + (0 / q2))
    a_par_r = tf.complex(one + (0 * q2) + (0 / q2), one + (0 * q2) + (0 / q2))
    a_perp_l = tf.complex(one + (0 * q2) + (0 / q2), one + (0 * q2) + (0 / q2))
    a_perp_r = tf.complex(one + (0 * q2) + (0 / q2), one + (0 * q2) + (0 / q2))
    a_zero_l = tf.complex(one + (0 * q2) + (0 / q2), one + (0 * q2) + (0 / q2))
    a_zero_r = tf.complex(one + (0 * q2) + (0 / q2), one + (0 * q2) + (0 / q2))

    j_1s = ((two + beta2_mu) / four)*(
        tf.math.abs(a_perp_l)**2 + tf.math.abs(a_par_l)**2 +
        tf.math.abs(a_perp_r)**2 + tf.math.abs(a_par_r)**2
    ) + four_mass_mu_over_q2*tf.math.real(
        a_perp_l * tf.math.conj(a_perp_r) +
        a_par_l * tf.math.conj(a_par_r)
    )

    j_1c = tf.math.abs(a_zero_l)**2 + tf.math.abs(a_zero_r)**2 + \
        four_mass_mu_over_q2*(2*tf.math.real(a_zero_l * tf.math.conj(a_zero_r)))

    j_2s = (beta2_mu / four)*(
        tf.math.abs(a_perp_l)**2 + tf.math.abs(a_par_l)**2 +
        tf.math.abs(a_perp_r)**2 + tf.math.abs(a_par_r)**2
    )

    j_2c = (- beta2_mu)*(tf.math.abs(a_zero_l)**2 + tf.math.abs(a_zero_r)**2)

    j_3 = (beta2_mu / two) * (
        tf.math.abs(a_perp_l) ** 2 - tf.math.abs(a_par_l) ** 2 +
        tf.math.abs(a_perp_r) ** 2 - tf.math.abs(a_par_r) ** 2
    )

    j_4 = (beta2_mu / tf.sqrt(two)) * (
        tf.math.real(a_zero_l * tf.math.conj(a_par_l)) +
        tf.math.real(a_zero_r * tf.math.conj(a_par_r))
    )

    j_5 = tf.sqrt(two) * beta_mu * (
        tf.math.real(a_zero_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_zero_r * tf.math.conj(a_perp_r))
    )

    j_6s = two * beta_mu * (
        tf.math.real(a_par_l * tf.math.conj(a_perp_l)) -
        tf.math.real(a_par_r * tf.math.conj(a_perp_r))
    )

    j_7 = tf.sqrt(two) * beta_mu * (
        tf.math.imag(a_zero_l * tf.math.conj(a_par_l)) -
        tf.math.imag(a_zero_r * tf.math.conj(a_par_r))
    )

    j_8 = (beta2_mu / tf.sqrt(two)) * (
        tf.math.imag(a_zero_l * tf.math.conj(a_perp_l)) +
        tf.math.imag(a_zero_r * tf.math.conj(a_perp_r))
    )

    j_9 = beta2_mu * (
        tf.math.imag(tf.math.conj(a_par_l) * a_perp_l) +
        tf.math.imag(tf.math.conj(a_par_r) * a_perp_r)
    )

    return (9 / (32 * math.pi)) * (
        (j_1s * sin2_theta_k) +
        (j_1c * cos2_theta_k) +
        (j_2s * sin2_theta_k * cos_2theta_l) +
        (j_2c * cos2_theta_k * cos_2theta_l) +
        (j_3 * sin2_theta_k * sin2_theta_l * cos_2phi) +
        (j_4 * sin_2theta_k * sin_2theta_l * cos_phi) +
        (j_5 * sin_2theta_k * sin_theta_l * cos_phi) +
        (j_6s * sin2_theta_k * cos_theta_l) +
        (j_7 * sin_2theta_k * sin_theta_l * sin_phi) +
        (j_8 * sin_2theta_k * sin_2theta_l * sin_phi) +
        (j_9 * sin_2theta_k * sin_2theta_l * sin_2phi)
    )


def generate_signal(signal_samples, options_num):
    q2_distribution = tfd.Uniform(low=q2_min, high=q2_max)
    cos_theta_k_distribution = tfd.Uniform(low=-1.0, high=1.0)
    cos_theta_l_distribution = tfd.Uniform(low=-1.0, high=1.0)
    phi_distribution = tfd.Uniform(low=-2*math.pi, high=2*math.pi)

    def _print(name, t):
        tf.print(name, "(shape", tf.shape(t), "):\n", t, output_stream=sys.stdout, end="\n\n")

    options = tf.stack(
        [
            q2_distribution.sample(options_num),
            cos_theta_k_distribution.sample(options_num),
            cos_theta_l_distribution.sample(options_num),
            phi_distribution.sample(options_num)
        ],
        axis=1,
        name='signal_options'
    )
    _print("options", options)

    decay_rates = decay_rate(options)
    _print("decay_rates", decay_rates)

    total_decay_rate = tf.reduce_sum(decay_rates)
    _print("total_decay_rate", total_decay_rate)

    probabilities = decay_rates / total_decay_rate
    _print("probabilities", probabilities)

    keys = np.random.choice(options.get_shape()[0], signal_samples, p=probabilities.numpy())
    _print("keys", keys)

    signal = tf.gather(options, keys)
    _print("signal", signal)

    return signal


s = generate_signal(10_000, 10_000_000)

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Signal distributions')
titles = [
    r'$q^2$',
    r'$\cos{\theta_k}$',
    r'$\cos{\theta_l}$',
    r'$\phi$'
]

for ax, feature, title in zip(axes.flatten(), s.numpy().transpose(), titles):
    sns.distplot(feature, ax=ax, bins=20)
    ax.set(title=title)

plt.show()
