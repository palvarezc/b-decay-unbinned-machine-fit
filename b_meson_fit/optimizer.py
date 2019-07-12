"""Class to do optimization"""
import tensorflow.compat.v2 as tf

import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs

tf.enable_v2_behavior()


class Optimizer:
    grads_timeline = []
    """Attributes:
        grads_timeline (list of tensors): History of gradients added to in each step
        step (int): Step counter
    """

    def __init__(
            self,
            script,
            fit_coeffs,
            signal_events,
            opt_name,
            signal_coeffs=None,
            **kwargs
    ):
        """Args:
            script (Script): Script context we're running under
            fit_coeffs (list of tensors): List of our coefficients we're fitting. Some can be tf.constant()s
            signal_events (tensor): Generated signal events
            opt_name (str): Name of the optimizer to use (e.g. Adam)
            signal_coeffs (list of tensors, optional): List of coefficients we're aiming for
            **kwargs: Additional args to pass to optimizer
         """
        self.script = script
        self.fit_coeffs = fit_coeffs
        self.trainables = bmfc.trainables(self.fit_coeffs)
        self.signal_events = signal_events

        self.signal_coeffs = signal_coeffs
        self.summary_writer = script.log.writer() if script.log else None
        self.signal_writer = script.log.writer('signal') if script.log and signal_coeffs else None

        self.optimizer = getattr(tf.optimizers, opt_name)(**kwargs)

        self.step = tf.Variable(0, name='global_step', dtype=tf.int64)

        self.grads_timeline = []
        self.latest_normalized_nll = self.normalized_nll()
        self.latest_grads = None
        self.latest_grad_mean = None
        self.latest_grad_total = None
        self.latest_grad_max = None

        self._write_summaries()

    def normalized_nll(self):
        """Get the normalized negative log likelihood

        Working with the normalised version ensures we don't need to re-optimize hyper-parameters when we
        change signal event numbers.

        Returns:
            Scalar tensor
        """
        return bmfs.normalized_nll(self.fit_coeffs, self.signal_events)

    def minimize(self):
        """Perform minimization step and write Tensorboard summaries if needed
        """
        self.step.assign(self.step + 1)

        [
            self.latest_normalized_nll,
            self.latest_grads,
            self.latest_grad_max,
            self.latest_grad_mean,
            self.latest_grad_total
        ] = self._do_gradients()
        self.grads_timeline.append(self.latest_grads)

        self._write_summaries()

    @tf.function
    def _do_gradients(self):
        with tf.device('/device:GPU:0'):
            with tf.GradientTape() as tape:
                normalized_nll = self.normalized_nll()
            grads = tape.gradient(normalized_nll, self.trainables)
            self.optimizer.apply_gradients(zip(grads, self.trainables))

            grad_max = tf.reduce_max(grads)
            grad_mean = tf.reduce_mean(grads)
            grad_total = tf.reduce_sum(grads)

            return [normalized_nll, grads, grad_max, grad_mean, grad_total]

    @tf.function
    def _write_summaries(self):
        # If the script context has `log` set to True then write our summaries
        if self.summary_writer:
            with self.summary_writer.as_default():
                # Macro scalars
                tf.summary.scalar('normalized_nll', self.latest_normalized_nll, step=self.step)
                if self.latest_grad_max:
                    tf.summary.scalar('gradients/max', self.latest_grad_max, step=self.step)
                if self.latest_grad_mean:
                    tf.summary.scalar('gradients/mean', self.latest_grad_mean, step=self.step)
                if self.latest_grad_total:
                    tf.summary.scalar('gradients/total', self.latest_grad_total, step=self.step)

                # All trainable coefficients and gradients as individual scalars
                for idx, coeff in enumerate(self.trainables):
                    name = bmfc.names[self.fit_coeffs.index(coeff)]
                    tf.summary.scalar('coefficients/' + name, coeff, step=self.step)
                    if self.latest_grads:
                        tf.summary.scalar('gradients/' + name, self.latest_grads[idx], step=self.step)

        # If the script context has `log` set to True AND we had `signal_coeffs` passed on creation,
        #  then log the true signal coefficient values for this step
        if self.signal_writer:
            with self.signal_writer.as_default():
                for coeff in self.trainables:
                    idx = self.fit_coeffs.index(coeff)
                    tf.summary.scalar('coefficients/' + bmfc.names[idx], self.signal_coeffs[idx], step=self.step)

    @staticmethod
    def log_signal_line(script, fit_coeffs, signal_coeffs, iterations):
        """Static method to generate constant signal lines for Tensorboard.

        Useful so that they can be compared to different optimizer runs

        This function doesn't really belong in the optimizer, but it's where all the other Tensorboard writing is
        done for the time being.
        """
        with script.log.writer('signal').as_default():
            for coeff in bmfc.trainables(fit_coeffs):
                idx = fit_coeffs.index(coeff)
                for i in range(iterations):
                    tf.summary.scalar('coefficients/' + bmfc.names[idx], signal_coeffs[idx], step=i)

            tf.summary.flush()

    def converged(self):
        """Has our optimizer converged on a solution

        Return:
            bool: Whether we've converged
        """
        # TODO: Implement this
        return False

    def print_step(self):
        """Output details about this step"""
        self.script.stdout(
            "Step:", self.step,
            "normalized_nll:", self.latest_normalized_nll,
            "grad_max:", self.latest_grad_max,
            "grad_mean:", self.latest_grad_mean,
            "grad_total:", self.latest_grad_total,
        )
        self.script.stdout("fit:   ", bmfc.to_str(self.fit_coeffs))
        if self.signal_coeffs:
            self.script.stdout("signal:", bmfc.to_str(self.signal_coeffs))
