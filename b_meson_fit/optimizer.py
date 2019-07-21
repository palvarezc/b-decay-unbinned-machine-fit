"""Class to do optimization"""
import numpy as np
import tensorflow.compat.v2 as tf

import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs

tf.enable_v2_behavior()


class Optimizer:
    def __init__(
            self,
            fit_coeffs,
            signal_events,
            opt_name='AMSGrad',
            learning_rate=0.20,
            opt_args=None,
            grad_clip=None,
            grad_max_cutoff=5e-7,
    ):
        """Args:
            fit_coeffs (list of tensors): List of our coefficients we're fitting. Some can be tf.constant()s
            signal_events (tensor): Generated signal events
            opt_name (str): Name of the optimizer to use (e.g. Adam). AMSGrad is a special value that sets the Adam
                optimizer with the amsgrad=True parameter
            learning_rate (float): Learning rate for optimizer
            opt_args (dict of str: str): Additional args to pass to optimizer
            grad_clip (float): Gradient global norm clipping value. Default only applied if `opt_name` is left
                as None
            grad_max_cutoff (float): Say we're converged when the max gradient is below this value
         """
        self.fit_coeffs = fit_coeffs

        self.trainables = self._remaining = bmfc.trainables(self.fit_coeffs)
        self.signal_events = signal_events

        if not opt_args:
            opt_args = {}
        opt_args = {'learning_rate': learning_rate, **opt_args}
        if opt_name == 'AMSGrad':
            opt_name = 'Adam'
            opt_args['amsgrad'] = True

        self.grad_clip = grad_clip
        self.grad_max_cutoff = grad_max_cutoff

        self.optimizer = getattr(tf.optimizers, opt_name)(**opt_args)

        self.step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)

        self.normalized_nll = self._normalized_nll()
        self.grads = None
        self.grad_max = None

    def minimize(self):
        """Increment step counter and calculate gradients"""
        self.step.assign(self.step + 1)
        self.normalized_nll, self.grads, self.grad_max = self._get_gradients()

    def converged(self):
        """Have all our coefficients finished training?

        Return:
            bool: Whether we've converged
        """
        if not self.grad_max:
            return False
        return self.grad_max.numpy() < self.grad_max_cutoff

    @tf.function
    def _get_gradients(self):
        """Calculate and apply gradients for this step"""
        with tf.GradientTape() as tape:
            normalized_nll = self._normalized_nll()
        grads = tape.gradient(normalized_nll, self.trainables)

        if self.grad_clip:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)

        self.optimizer.apply_gradients(zip(grads, self.trainables))

        return normalized_nll, grads, tf.math.abs(tf.reduce_max(grads))

    def _normalized_nll(self):
        """Get the normalized negative log likelihood

        Working with the normalised version ensures we don't need to re-optimize hyper-parameters when we
        change signal event numbers.

        Returns:
            Scalar tensor
        """
        return bmfs.normalized_nll(self.fit_coeffs, self.signal_events)
