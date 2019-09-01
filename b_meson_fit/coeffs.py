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
            [1.0, 0.0, 0.0]
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
            [-0.11366033229864046, 0.009293559978293, -0.04761546602270795]
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
            [1.0, 0.0, 0.0]
        ],
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ],
    ],
}
_signal_coeffs_p_wave_idxs = range(0, 36)
_signal_coeffs_s_wave_idxs = range(36, 48)
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

# Fit coefficient initialization schemes
FIT_INIT_TWICE_LARGEST_SIGNAL_SAME_SIGN = 'TWICE_LARGEST_SIGNAL_SAME_SIGN'
FIT_INIT_TWICE_CURRENT_SIGNAL_ANY_SIGN = 'TWICE_CURRENT_SIGNAL_ANY_SIGN'
FIT_INIT_CURRENT_SIGNAL = 'CURRENT_SIGNAL'
fit_init_schemes = [
    FIT_INIT_TWICE_LARGEST_SIGNAL_SAME_SIGN,
    FIT_INIT_TWICE_CURRENT_SIGNAL_ANY_SIGN,
    FIT_INIT_CURRENT_SIGNAL,
]
fit_initialization_scheme_default = FIT_INIT_TWICE_LARGEST_SIGNAL_SAME_SIGN


def signal(model):
    """Turn our signal coefficient numbers into a flat list of constant tensors
    """
    if model not in _signal_coeffs:
        raise ValueError('No {} signal coefficients defined'.format(model))
    return [tf.constant(_p) for _a in _signal_coeffs[model] for _c in _a for _p in _c]


def fit(initialization=fit_initialization_scheme_default, current_signal_model=None, fix_p_wave_model=None):
    """Construct a flat list of tensors to represent what we're going to fit

    Tensors that represent non-fixed coefficients in this basis are tf.Variables
    Tensors that represent fixed coefficients in this basis set as constant 0's

    Args:
        initialization (float or str): Constant float value to initialize all trainable coefficients to, or an
            initialization scheme listed in `fit_init_schemes`
        current_signal_model (str, optional): Current signal model. Must be specified for randomization if the
            FIT_INIT_TWICE_CURRENT_SIGNAL_ANY_SIGN or FIT_INIT_CURRENT_SIGNAL initialization schemes are used
        fix_p_wave_model (str, optional): Model to fix all P-wave coefficients to. Useful for generating Q test stats
    """
    if initialization in [FIT_INIT_TWICE_CURRENT_SIGNAL_ANY_SIGN, FIT_INIT_CURRENT_SIGNAL] \
            and current_signal_model is None:
        raise ValueError('initialization_model must be supplied when using {} initialization'.format(initialization))
    if current_signal_model is not None and current_signal_model not in signal_models:
        raise ValueError('current_signal_model {} unknown'.format(current_signal_model))
    current_signal_coeffs = signal(current_signal_model) if current_signal_model is not None else None
    if fix_p_wave_model is not None and fix_p_wave_model not in signal_models:
        raise ValueError('fix_p_wave_model {} unknown'.format(fix_p_wave_model))
    fix_p_wave_coeffs = signal(fix_p_wave_model) if fix_p_wave_model is not None else None

    max_signal_coeffs = [0.0] * count
    for signal_model in signal_models:
        signal_coeffs = [_p for _a in _signal_coeffs[signal_model] for _c in _a for _p in _c]
        for signal_idx, signal_coeff in enumerate(signal_coeffs):
            if abs(signal_coeff) > max_signal_coeffs[signal_idx]:
                max_signal_coeffs[signal_idx] = signal_coeff

    fit_trainable_ids = list(itertools.chain(range(0, 21), range(24, 27), [36], [39], [42], [45]))

    fit_coeffs = []

    for i in range(count):
        if i <= _signal_coeffs_p_wave_idxs[-1] and fix_p_wave_coeffs is not None:
            # This is a P-wave coeff and we've been told to copy a signal value
            coeff = tf.constant(fix_p_wave_coeffs[i])
        elif i in fit_trainable_ids:
            if initialization == FIT_INIT_TWICE_LARGEST_SIGNAL_SAME_SIGN:
                # Initialize coefficient from 0 to 2x largest value in all signal models
                init_value = tf.random.uniform(
                    shape=(),
                    minval=tf.math.minimum(0.0, 2 * max_signal_coeffs[i]),
                    maxval=tf.math.maximum(0.0, 2 * max_signal_coeffs[i]),
                )
            elif initialization == FIT_INIT_TWICE_CURRENT_SIGNAL_ANY_SIGN:
                # Initialize coefficient from -2x to +2x value in this signal model
                init_value = tf.random.uniform(
                    shape=(),
                    minval=-tf.math.abs(2 * current_signal_coeffs[i]),
                    maxval=tf.math.abs(2 * current_signal_coeffs[i]),
                )
            elif initialization == FIT_INIT_CURRENT_SIGNAL:
                # Initialize coefficient to the value in this signal model
                init_value = current_signal_coeffs[i]
            elif isinstance(initialization, float):
                # Initialize coefficient to the specified value (Useful for testing)
                init_value = initialization
            else:
                raise ValueError('Initialization scheme {} is undefined'.format(initialization))

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
