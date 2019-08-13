"""Contains relativistic Breit-Wigner functions"""
import tensorflow.compat.v2 as tf

import b_meson_fit.integrate as bmfi

tf.enable_v2_behavior()

# See p45 of 2018 PDG - K*0(700)
mass_k700 = tf.constant(0.824)  # 824 MeV
decay_width_k700 = tf.constant(0.478)  # 478 MeV
# See p868 of 2012 PDG (Used in original paper) - K*0(800)
# # mass_k700 = tf.constant(0.682)  # 824 MeV
# # decay_width_k700 = tf.constant(0.547)  # 478 MeV

# See p46 of 2018 PDG - K*0(892)
mass_k892 = tf.constant(0.89555)  # 895.55 MeV
decay_width_k892 = tf.constant(0.0473)  # 47.3 MeV
# See p870 of 2012 PDG (Used in original paper) - K*0(892)
# # mass_k892 = tf.constant(0.89594)  # 895.55 MeV
# # decay_width_k892 = tf.constant(0.0487)  # 47.3 MeV

mass_k_plus = tf.constant(0.493677)  # 493.677 MeV
mass_pi_minus = tf.constant(0.13957018)  # 139.57018 MeV

integration_max = tf.constant(100.0)
integration_dt = tf.constant(0.00025)


def k700_distribution(mass):
    """Calculate normalized relativistic Breit-Wigner distribution value for K(700) at given mass"""
    if k700_distribution.norm is None:
        k700_distribution.norm = _norm(_k700_distribution_unnormalized)
    return _k700_distribution_unnormalized(mass) / k700_distribution.norm


def k700_distribution_integrated():
    """Integrate normalized relativistic Breit-Wigner distribution for K(700) around K(892) mass"""
    return _integrate_distribution_around_k892(k700_distribution)


def k892_distribution(mass):
    """Calculate normalized relativistic Breit-Wigner distribution value for K(892) at given mass"""
    if k892_distribution.norm is None:
        k892_distribution.norm = _norm(_k892_distribution_unnormalized)
    return _k892_distribution_unnormalized(mass) / k892_distribution.norm


def k892_distribution_integrated():
    """Integrate normalized relativistic Breit-Wigner distribution for K(892) around K(892) mass"""
    return _integrate_distribution_around_k892(k892_distribution)


def k700_k892_distribution(mass):
    """Calculate normalized relativistic Breit-Wigner distribution value for K(700)/K(892) mix at given mass"""
    if k700_k892_distribution.norm is None:
        k700_k892_distribution.norm = _norm(_k700_k892_distribution_unnormalized)
    return _k700_k892_distribution_unnormalized(mass) / tf.cast(k700_k892_distribution.norm, dtype=tf.complex64)


def k700_k892_distribution_integrated():
    """Integrate normalized relativistic Breit-Wigner distribution for K(700)/K(892 around K(892) mass"""
    return _integrate_distribution_around_k892(k700_k892_distribution)


# Values of norm of distributions. These will be set on the first use and stored in these variables
#  so that is it not calculated on every point of a BW plot
k700_distribution.norm = None
k892_distribution.norm = None
k700_k892_distribution.norm = None


def _k700_distribution_unnormalized(mass):
    return tf.math.abs(_k700_amplitude(mass)) ** 2


def _k892_distribution_unnormalized(mass):
    return tf.math.abs(_k892_amplitude(mass)) ** 2


def _k700_k892_distribution_unnormalized(mass):
    return _k700_amplitude(mass) * tf.math.conj(_k892_amplitude(mass))


def _k700_amplitude(mass):
    return _amplitude(mass, 0, mass_k700, decay_width_k700)


def _k892_amplitude(mass):
    return _amplitude(mass, 1, mass_k892, decay_width_k892)


def _integrate_distribution_around_k892(distribution):
    """Integrate distribution between +/- 100 MeV of K892 mass"""
    return bmfi.trapezoid(distribution, mass_k892 - 0.1, mass_k892 + 0.1, integration_dt)


def _norm(distribution_unnormalized):
    """
    Integrate distribution over all values

    Use min of K+/Pi- system mass as momentum is undefined below this
    """
    return bmfi.trapezoid(distribution_unnormalized, mass_k_plus + mass_pi_minus, integration_max, integration_dt)


def _amplitude(mass, l, energy, decay_width_0):
    """
    Calculate relativistic Breit-Wigner amplitude at given mass

    Args:
        mass (tensor): Rank-0 tensor of mass
        l (int): Angular momentum (1 for P-wave, 0 for S-wave)
        energy (tensor): Rank-0 tensor of parent K mass
        decay_width_0 (tensor): Rank-0 tensor of parent K particle with no mass dependence
    """
    def _q(_mass):  # Momentum of K+/Pi- daughters in rest frame of resonance
        return (
            tf.sqrt(
                ((_mass ** 2) - ((mass_k_plus + mass_pi_minus) ** 2))
                *
                ((_mass ** 2) - ((mass_k_plus - mass_pi_minus) ** 2))
            )
        ) / (
            2.0 * _mass
        )

    q = _q(mass)
    q_0 = _q(energy)

    if l == 1:
        # P-wave
        r = tf.constant(4.0)  # Radius of hadron. 4 Gev^-1 / ~ 0.8 fm
        z = tf.math.abs(q) * r
        z_0 = tf.math.abs(q_0) * r
        barrier = tf.sqrt((1.0 + (z_0 ** 2)) / (1.0 + (z ** 2)))
    elif l == 0:
        # S-wave
        barrier = tf.constant(1.0)
    else:
        raise ValueError('l must be 0 or 1. Found: {}'.format(l))

    decay_width_mass_dependent = decay_width_0 * ((q / q_0) ** ((2 * l) + 1)) * (energy / mass) * (barrier ** 2)

    return 1.0 / tf.complex((energy ** 2) - (mass ** 2), -energy * decay_width_mass_dependent)
