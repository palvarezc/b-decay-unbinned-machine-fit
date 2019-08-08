"""
Contains constants and methods for handling signal and fit coefficients

Fit and signal are in the fixed basis where Im(A_perp_r) = Im(A_zero_l) = Re(A_zero_r) = Im(A_zero_r) = 0.
"""
import itertools
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

# Models for our signal coeffs
SM = 'SM'
NP = 'NP'

# Signal coefficients generated from flavio thanks to Mark Smith of Imperial
# Outer arrays: [a_par_l, a_par_r, a_perp_l, a_perp_r, a_0_l, a_0_r, a_00_l, a_00_r]
# Inner arrays: [Re(...), Im(...)
# Inner array coeffs: [a, b, c] for anzatz a + (b * q2) + (c / q2)
_signal_coeffs = {
    SM: [
        [
            [-4.178102308587205, -0.15184343863801822, 6.818324577344573],
            [0.008585377715927508, -0.001823001658320108, 0.46607419549466444]
        ],
        [
            [-0.23538124837356975, -0.004317631254342448, 8.00374551265976],
            [0.16564202403696493, -0.013095878427794742, -0.3066801644683403]
        ],
        [
            [3.886406712838685, 0.08526550962215246, -8.197445982405629],
            [-0.09505176167285938, 0.007934013042069043, -0.07297003098318804]
        ],
        [
            [-0.4235836013545314, 0.027298994876497513, -7.147450839543464],
            [0.0, 0.0, 0.0]
        ],
        [
            [7.202758939554573, -0.2278163014848678, 9.898629947119229],
            [0.0, 0.0, 0.0]
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ],
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ],
    ],
    NP: [  # C9 = -1.027, C10 = 0.498
        [
            [-3.4277495848061257, -0.12410026985551571, 6.045281152442963],
            [0.00934061365013997, -0.001989193837745718, 0.5034113300277555]
        ],
        [
            [-0.25086978961912654, -0.005180213333933305, 8.636744983192575],
            [0.2220926359265556, -0.017419352926410284, -0.528067287659531]
        ],
        [
            [3.0646407176207813, 0.07851536717584778, -8.841144517240298],
            [-0.11366033229864046, 0.009293559978293, 0.04761546602270795]
        ],
        [
            [-0.9332669880450042, 0.01686711151445955, -6.318555350023665],
            [0.0, 0.0, 0.0]
        ],
        [
            [5.882883042792871, -0.18442496620391777, 8.10139804649606],
            [0.0, 0.0, 0.0]
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ],
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ],
    ],
}
signal_models = list(_signal_coeffs.keys())

amplitude_names = [
    'a_para_l_re',
    'a_para_l_im',
    'a_para_r_re',
    'a_para_r_im',
    'a_perp_l_re',
    'a_perp_l_im',
    'a_perp_r_re',
    'a_perp_r_im',
    'a_0_l_re',
    'a_0_l_im',
    'a_0_r_re',
    'a_0_r_im',
    'a_00_l_re',
    'a_00_l_im',
    'a_00_r_re',
    'a_00_r_im',
]
amplitude_latex_names = [
    r'Re($a_{\parallel}^L$)',
    r'Im($a_{\parallel}^L$)',
    r'Re($a_{\parallel}^R$)',
    r'Im($a_{\parallel}^R$)',
    r'Re($a_{\bot}^L$)',
    r'Im($a_{\bot}^L$)',
    r'Re($a_{\bot}^R$)',
    r'Im($a_{\bot}^R$)',
    r'Re($a_{0}^L$)',
    r'Im($a_{0}^L$)',
    r'Re($a_{0}^R$)',
    r'Im($a_{0}^R$)',
    r'Re($a_{00}^L$)',
    r'Im($a_{00}^L$)',
    r'Re($a_{00}^R$)',
    r'Im($a_{00}^R$)',
]
amplitude_count = len(amplitude_names)

param_names = ['alpha', 'beta', 'gamma']
param_latex_names = [r'$\alpha$', r'$\beta$', r'$\gamma$']
param_count = len(param_names)

names = ['{}_{}'.format(a, p) for a in amplitude_names for p in param_names]
latex_names = ['{} {}'.format(a, p) for a in amplitude_latex_names for p in param_latex_names]
count = len(names)

# If fit_default is not set, then fit coefficients should be randomly generated (See fit())
fit_default = None


def signal(name):
    """Turn our signal coefficient numbers into a flat list of constant tensors
    """
    if name not in _signal_coeffs:
        raise ValueError('No {} signal coefficients defined'.format(name))
    return [tf.constant(_p) for _a in _signal_coeffs[name] for _c in _a for _p in _c]


def fit(signal_coeffs=None):
    """Construct a flat list of tensors to represent what we're going to fit.

    If the global fit_default is set as None then generate random initial values
    of +/- 100% of the signal value for this coefficient.

    Tensors that represent non-fixed coefficients in this basis are tf.Variables
    Tensors that represent fixed coefficients in this basis set as constant 0's

    Args:
        signal_coeffs (list of tensors, optional): If randomization is enabled then the signal
            coeffs must be passed in
    """
    fit_trainable_ids = list(itertools.chain(range(0, 21), range(24, 27), [36], [39], [42], [45]))

    fit_coeffs = []

    for i in range(count):
        if i in fit_trainable_ids:
            if fit_default:
                init_value = fit_default
            else:
                if signal_coeffs is None:
                    raise ValueError('signal_coeffs must be specified with fit coefficient randomization')
                init_value = tf.random.uniform(
                    shape=(),
                    minval=tf.math.minimum(0.0, 2 * signal_coeffs[i]),
                    maxval=tf.math.maximum(0.0, 2 * signal_coeffs[i]),
                )

            coeff = tf.Variable(init_value, name=names[i])
        else:
            coeff = tf.constant(0.0)
        fit_coeffs.append(coeff)

    return fit_coeffs


def trainables(coeffs):
    """Get sublist of coefficients that are trainable"""
    return [c for c in coeffs if is_trainable(c)]


def is_trainable(coeff):
    """
    Check if a coefficient is trainable. This is done by seeing if the tensor has the 'trainable'
    attribute and it is True. tf.constant's do not have this attribute
    """
    return getattr(coeff, 'trainable', False)


def to_str(coeffs):
    """
    Take a list of coefficients and return a single-line string for printing
    Coefficients that are constants are marked with a trailing '*'
    """
    c_list = []
    for c in coeffs:
        c_list.append('{:5.2f}{}'.format(c.numpy(), ' ' if is_trainable(c) else '*'))
    return ' '.join(c_list)
