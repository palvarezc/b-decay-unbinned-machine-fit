import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

with tf.device('/device:GPU:0'):
    # Signal generated from flavio for C9 = -1.027, C10 = 0.498
    signal_coeffs = [
        tf.constant(-3.4277495848061257, name='sig_para_l_re_alpha'),
        tf.constant(-0.12410026985551571, name='sig_para_l_re_beta'),
        tf.constant(6.045281152442963, name='sig_para_l_re_gamma'),
        tf.constant(0.00934061365013997, name='sig_para_l_im_alpha'),
        tf.constant(-0.001989193837745718, name='sig_para_l_im_beta'),
        tf.constant(0.5034113300277555, name='sig_para_l_im_gamma'),

        tf.constant(-0.25086978961912654, name='sig_para_r_re_alpha'),
        tf.constant(-0.005180213333933305, name='sig_para_r_re_beta'),
        tf.constant(8.636744983192575, name='sig_para_r_re_gamma'),
        tf.constant(0.2220926359265556, name='sig_para_r_im_alpha'),
        tf.constant(-0.017419352926410284, name='sig_para_r_im_beta'),
        tf.constant(-0.528067287659531, name='sig_para_r_im_gamma'),

        tf.constant(3.0646407176207813, name='sig_perp_l_re_alpha'),
        tf.constant(0.07851536717584778, name='sig_perp_l_re_beta'),
        tf.constant(-8.841144517240298, name='sig_perp_l_re_gamma'),
        tf.constant(-0.11366033229864046, name='sig_perp_l_im_alpha'),
        tf.constant(0.009293559978293, name='sig_perp_l_im_beta'),
        tf.constant(0.04761546602270795, name='sig_perp_l_im_gamma'),

        tf.constant(-0.9332669880450042, name='sig_perp_r_re_alpha'),
        tf.constant(0.01686711151445955, name='sig_perp_r_re_beta'),
        tf.constant(-6.318555350023665, name='sig_perp_r_re_gamma'),
        tf.constant(0.0, name='sig_perp_r_im_alpha'),
        tf.constant(0.0, name='sig_perp_r_im_beta'),
        tf.constant(0.0, name='sig_perp_r_im_gamma'),

        tf.constant(5.882883042792871, name='sig_zero_l_re_alpha'),
        tf.constant(-0.18442496620391777, name='sig_zero_l_re_beta'),
        tf.constant(8.10139804649606, name='sig_zero_l_re_gamma'),
        tf.constant(0.0, name='sig_zero_l_im_alpha'),
        tf.constant(0.0, name='sig_zero_l_im_beta'),
        tf.constant(0.0, name='sig_zero_l_im_gamma'),

        tf.constant(0.0, name='sig_zero_r_re_alpha'),
        tf.constant(0.0, name='sig_zero_r_re_beta'),
        tf.constant(0.0, name='sig_zero_r_re_gamma'),
        tf.constant(0.0, name='sig_zero_r_im_alpha'),
        tf.constant(0.0, name='sig_zero_r_im_beta'),
        tf.constant(0.0, name='sig_zero_r_im_gamma'),
    ]

    params = ['alpha', 'beta', 'gamma']
    fit_coeffs = \
        [tf.Variable(1.0, name='fit_a_para_l_re_{}'.format(p)) for p in params] + \
        [tf.Variable(1.0, name='fit_a_para_l_im_{}'.format(p)) for p in params] + \
        [tf.Variable(1.0, name='fit_a_para_r_re_{}'.format(p)) for p in params] + \
        [tf.Variable(1.0, name='fit_a_para_r_im_{}'.format(p)) for p in params] + \
        [tf.Variable(1.0, name='fit_a_perp_l_re_{}'.format(p)) for p in params] + \
        [tf.Variable(1.0, name='fit_a_perp_l_im_{}'.format(p)) for p in params] + \
        [tf.Variable(1.0, name='fit_a_perp_r_re_{}'.format(p)) for p in params] + \
        [tf.constant(0.0, name='fit_a_perp_r_im_{}'.format(p)) for p in params] + \
        [tf.Variable(1.0, name='fit_a_zero_l_re_{}'.format(p)) for p in params] + \
        [tf.constant(0.0, name='fit_a_zero_l_im_{}'.format(p)) for p in params] + \
        [tf.constant(0.0, name='fit_a_zero_r_re_{}'.format(p)) for p in params] + \
        [tf.constant(0.0, name='fit_a_zero_r_im_{}'.format(p)) for p in params]

trainable_coeffs = [c for c in fit_coeffs if getattr(c, 'trainable', False)]


def coeffs_to_string(coeffs):
    coeffs_strs = []
    for coeff in coeffs:
        coeffs_strs.append('{}{:5.2f}'.format(' ' if getattr(coeff, 'trainable', False) else '=', coeff.numpy()))
    return ' '.join(coeffs_strs)