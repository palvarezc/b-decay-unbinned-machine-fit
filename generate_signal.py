import math
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

tf.enable_v2_behavior()
tfd = tfp.distributions


def decay_rate(x):
    q2 = x[0]
    cos_theta_k = x[1]
    cos_theta_l = x[2]
    phi = x[3]

    cos2_theta_k = tf.square(cos_theta_k)
    sin2_theta_k = tf.subtract(1.0, cos2_theta_k)

    a_par_l = tf.complex(1.0 + 0 * q2 + 0 / q2, 1.0 + 0 * q2 + 0 / q2)
    a_par_r = tf.complex(1.0 + 0 * q2 + 0 / q2, 1.0 + 0 * q2 + 0 / q2)
    a_perp_l = tf.complex(1.0 + 0 * q2 + 0 / q2, 1.0 + 0 * q2 + 0 / q2)
    a_perp_r = tf.complex(1.0 + 0 * q2 + 0 / q2, 1.0 + 0 * q2 + 0 / q2)
    a_zero_l = tf.complex(1.0 + 0 * q2 + 0 / q2, 1.0 + 0 * q2 + 0 / q2)
    a_zero_r = tf.complex(1.0 + 0 * q2 + 0 / q2, 1.0 + 0 * q2 + 0 / q2)

    j_1s = tf.math.real(a_par_l)

    return (9 / (32 * math.pi)) * (
        tf.multiply(j_1s, sin2_theta_k)
    )


def generate_signal(signal_samples, variable_samples=10000):
    cos_theta_k_distribution = tfd.Uniform(low=-1.0, high=1.0)
    cos_theta_l_distribution = tfd.Uniform(low=-1.0, high=1.0)
    phi_distribution = tfd.Uniform(low=-2 * math.pi, high=2 * math.pi)
    q2_distribution = tfd.Uniform(low=2.0, high=6.0)

    def _print(name, t):
        tf.print(name, "(shape", tf.shape(t), "):\n", t, output_stream=sys.stdout, end="\n\n")

    # TODO: Is number of options right?
    options = tf.stack(
        [
            q2_distribution.sample(variable_samples),
            cos_theta_k_distribution.sample(variable_samples),
            cos_theta_l_distribution.sample(variable_samples),
            phi_distribution.sample(variable_samples)
        ],
        axis=1,
        name='signal_options'
    )
    _print("options", options)

    decay_rates = tf.map_fn(lambda x: decay_rate(x), options)
    _print("decay_rates", decay_rates)

    total_decay_rate = tf.reduce_sum(decay_rates)
    _print("total_decay_rate", total_decay_rate)

    probabilities = tf.map_fn(lambda x: x / total_decay_rate, decay_rates)
    _print("probabilities", probabilities)

    keys = np.random.choice(options.get_shape()[0], signal_samples, p=probabilities.numpy())
    _print("keys", keys)

    # FIXME: Do not convert to numpy
    signal = np.take(options.numpy().transpose(), keys, axis=1).transpose()
    _print("signal", signal)

    return signal


s = generate_signal(10000)

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.suptitle('Sample distributions')
titles = [
    r'$q^2$',
    r'$\cos{\theta_k}$',
    r'$\cos{\theta_l}$',
    r'$\phi$'
]

for ax, feature, name in zip(axes.flatten(), s.transpose(), titles):
    sns.distplot(feature, ax=ax, bins=20)
    ax.set(title=name)

plt.show()

exit(0)
