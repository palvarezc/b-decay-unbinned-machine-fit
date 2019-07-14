"""Class to do optimization"""
import numpy as np
import tensorflow.compat.v2 as tf

import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs

tf.enable_v2_behavior()


class Optimizer:
    default_opt_name = 'Adam'
    # Increasing epsilon from default 1e-8 helps stop dithering at higher iterations
    default_opt_args = {'learning_rate': 0.10, 'epsilon': 1e-7}
    default_grad_clip = None
    default_grad_cutoff_count = 100
    default_grad_cutoff_value = 5e-7

    def __init__(
            self,
            fit_coeffs,
            signal_events,
            opt_name=None,
            opt_args=None,
            grad_clip=None,
            grad_cutoff=True,
            grad_cutoff_count=None,
            grad_cutoff_value=None
    ):
        """Args:
            fit_coeffs (list of tensors): List of our coefficients we're fitting. Some can be tf.constant()s
            signal_events (tensor): Generated signal events
            opt_name (str): Name of the optimizer to use (e.g. Adam). Unless this is left as the default, param
                and clipping defaults will not be applied.
            opt_args (dict of str: str): Additional args to pass to optimizer
            grad_clip (float): Gradient global norm clipping value. Default only applied if `opt_name` is left
                as None
            grad_cutoff (bool): Whether if stop training certain variables when we think they've converged.
            grad_cutoff_count (int): Say we're converged when the gradient stddev has been below a value for this many
                steps
            grad_cutoff_value (float): Say we're converged when the gradient stddev has been below this value for a
                number of steps
         """
        self.fit_coeffs = fit_coeffs
        self.trainables = self._remaining = bmfc.trainables(self.fit_coeffs)
        self.train_mask = tf.fill([len(self.trainables)], True)
        self.signal_events = signal_events

        if opt_args and not opt_name:
            raise ValueError('opt_name must be specified if opt_args are')

        self.grad_clip = grad_clip
        if not opt_name:
            opt_name = self.default_opt_name
            opt_args = self.default_opt_args
            if not grad_clip:
                self.grad_clip = self.default_grad_clip

        self.grad_cutoff = grad_cutoff
        if not grad_cutoff_count:
            self.grad_cutoff_count = self.default_grad_cutoff_count
        if not grad_cutoff_value:
            self.grad_cutoff_value = self.default_grad_cutoff_value

        self.optimizer = getattr(tf.optimizers, opt_name)(**opt_args)

        self.step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)

        self.normalized_nll = self._normalized_nll()
        self.grads = None

        self._timeline_grads = tf.zeros([0, 24])

    def minimize(self):
        """Increment step counter, calculate gradients, and see if any coeffs have converged"""
        self.step.assign(self.step + 1)

        # Get a list of coefficients that still need training
        to_train = np.array(self.trainables)[self.train_mask.numpy()].tolist()

        self.normalized_nll, sparse_grads = self._get_gradients(to_train)

        # Our list of gradients from _get_gradients() only contains coefficients we're still training.
        # Expand it out to cover all coefficients by inserting 0.0 for coefficients that have
        #  finished training
        zero = tf.constant(0.0)
        grads_iter = iter(sparse_grads)
        self.grads = [next(grads_iter) if i else zero for i in self.train_mask.numpy()]

        if self.grad_cutoff:
            # Add these gradients to our timeline
            self._timeline_grads = tf.concat([[self.grads], self._timeline_grads], 0)

            # If our gradient timeline is full (I.e. it's got grad_cutoff_count` rows in it)
            if self.step.numpy() % self.grad_cutoff_count == 0:
                # Work out which coefficients are still training by seeing which ones have a stddev
                #  of > `grad_cutoff_value`
                self.train_mask = tf.greater(
                    tf.math.abs(tf.math.reduce_std(self._timeline_grads, axis=0)),
                    self.grad_cutoff_value
                )
                # Reset our timeline
                self._timeline_grads = tf.zeros([0, 24])

    def converged(self):
        """Have all our coefficients finished training?

        Return:
            bool: Whether we've converged
        """
        any_remaining = tf.cast(tf.math.count_nonzero(self.train_mask), dtype=tf.bool).numpy()
        return not any_remaining

    @tf.function
    def _get_gradients(self, to_train):
        """Calculate and apply gradients for this step"""
        with tf.GradientTape() as tape:
            normalized_nll = self._normalized_nll()
        grads = tape.gradient(normalized_nll, to_train)

        if self.grad_clip:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)

        self.optimizer.apply_gradients(zip(grads, to_train))

        return normalized_nll, grads

    def _normalized_nll(self):
        """Get the normalized negative log likelihood

        Working with the normalised version ensures we don't need to re-optimize hyper-parameters when we
        change signal event numbers.

        Returns:
            Scalar tensor
        """
        return bmfs.normalized_nll(self.fit_coeffs, self.signal_events)
