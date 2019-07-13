"""Class to do optimization"""
import tensorflow.compat.v2 as tf

import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs

tf.enable_v2_behavior()


class Optimizer:
    default_opt_name = 'Adam'
    default_opt_args = {'learning_rate': 0.20}
    default_grad_clip = 5.0
    default_grad_cutoff_count = 200
    default_grad_cutoff_value = 4e-6

    def __init__(
            self,
            fit_coeffs,
            signal_events,
            opt_name=None,
            opt_args=None,
            grad_clip=None,
            grad_cutoff_count=None,
            grad_cutoff_value=None
    ):
        """Args:
            fit_coeffs (list of tensors): List of our coefficients we're fitting. Some can be tf.constant()s
            signal_events (tensor): Generated signal events
            opt_name (str): Name of the optimizer to use (e.g. Adam). Unless this is left as the default, no
                other defaults will be applied and will need tuning manually.
            opt_args (dict of str: str): Additional args to pass to optimizer
            grad_clip (float): Gradient global norm clipping value
            grad_cutoff_count (int): Say we're converged when the max gradient has been below a value for this many
                steps
            grad_cutoff_value (float): Say we're converged when the max gradient has been below this value for a
                number of steps
         """
        self.fit_coeffs = fit_coeffs
        self.trainables = bmfc.trainables(self.fit_coeffs)
        self.signal_events = signal_events

        if opt_args and not opt_name:
            raise ValueError('opt_name must be specified if opt_args are')

        self.grad_clip = grad_clip
        if not opt_name:
            opt_name = self.default_opt_name
            opt_args = self.default_opt_args
            if not grad_clip:
                self.grad_clip = self.default_grad_clip
            if not grad_cutoff_count:
                self.grad_cutoff_count = self.default_grad_cutoff_count
            if not grad_cutoff_value:
                self.grad_cutoff_value = self.default_grad_cutoff_value

        self.optimizer = getattr(tf.optimizers, opt_name)(**opt_args)

        self.step = tf.Variable(0, name='global_step', dtype=tf.int64)

        self.normalized_nll = self._normalized_nll()
        self.grad_max = None
        self.grads = None
        self.timeline_grad_max = []

    def minimize(self):
        """Perform minimization step and increment step counter"""
        self.step.assign(self.step + 1)
        self.normalized_nll, self.grad_max, self.grads = self._do_gradients()
        self.timeline_grad_max.append(self.grad_max)

    def converged(self):
        """Has our optimizer converged on a solution?

        Stop if the max gradient has been less `grad_cutoff_value` for `grad_cutoff_count` steps

        Return:
            bool: Whether we've converged
        """
        if len(self.timeline_grad_max) < self.grad_cutoff_count:
            return False

        if tf.math.reduce_max(self.timeline_grad_max[-self.grad_cutoff_count:]) < self.grad_cutoff_value:
            return True

        return False

    @tf.function
    def _do_gradients(self):
        """Calculate and apply gradients for this step"""
        with tf.device('/device:GPU:0'):
            with tf.GradientTape() as tape:
                normalized_nll = self._normalized_nll()
            grads = tape.gradient(normalized_nll, self.trainables)
            if self.grad_clip:
                grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
            self.optimizer.apply_gradients(zip(grads, self.trainables))

            grad_max = tf.reduce_max(grads)

        return normalized_nll, grad_max, grads

    def _normalized_nll(self):
        """Get the normalized negative log likelihood

        Working with the normalised version ensures we don't need to re-optimize hyper-parameters when we
        change signal event numbers.

        Returns:
            Scalar tensor
        """
        return bmfs.normalized_nll(self.fit_coeffs, self.signal_events)
