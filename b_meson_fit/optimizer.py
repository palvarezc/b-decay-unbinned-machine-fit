"""Class to do optimization"""
import tensorflow.compat.v2 as tf

import b_meson_fit.coeffs as bmfc
import b_meson_fit.signal as bmfs

tf.enable_v2_behavior()


class Optimizer:
    grads_timeline = []
    step = 0
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
        self.signal_events = signal_events

        self.signal_coeffs = signal_coeffs
        self.summary_writer = script.log.writer() if script.log else None
        self.signal_writer = script.log.writer('signal') if script.log and signal_coeffs else None

        self.optimizer = getattr(tf.optimizers, opt_name)(**kwargs)

        self.latest_nll = self.nll()
        self.latest_grads = None
        self.latest_grad_mean = None
        self.latest_grad_total = None
        self.latest_grad_max = None

        self._write_summaries()

    def nll(self):
        """Get the negative log likelihood for our coefficients and signal events

        Returns:
            Scalar tensor
        """
        return bmfs.nll(self.fit_coeffs, self.signal_events)

    def minimize(self):
        """Perform minimization step and write Tensorboard summaries if needed
        """
        self.step = self.step + 1
        with tf.device('/device:GPU:0'):
            with tf.GradientTape() as tape:
                self.latest_nll = self.nll()
            trainables = bmfc.trainables(self.fit_coeffs)
            grads = tape.gradient(self.latest_nll, trainables)
            self.optimizer.apply_gradients(zip(grads, trainables))

            self.grads_timeline.append(grads)
            self.latest_grads = grads
            self.latest_grad_max = tf.reduce_max(grads)
            self.latest_grad_mean = tf.reduce_mean(grads)
            self.latest_grad_total = tf.reduce_sum(grads)

            self._write_summaries()

    def _write_summaries(self,):
        # If the script context has `log` set to True then write our summaries
        if self.summary_writer:
            with self.summary_writer.as_default():
                # Macro scalars
                tf.summary.scalar('nll', self.latest_nll, step=self.step)
                if self.latest_grad_max:
                    tf.summary.scalar('gradients/max', self.latest_grad_max, step=self.step)
                if self.latest_grad_mean:
                    tf.summary.scalar('gradients/mean', self.latest_grad_mean, step=self.step)
                if self.latest_grad_total:
                    tf.summary.scalar('gradients/total', self.latest_grad_total, step=self.step)

                # All trainable coefficients and gradients as individual scalars
                for idx, coeff in enumerate(bmfc.trainables(self.fit_coeffs)):
                    name = bmfc.names[self.fit_coeffs.index(coeff)]
                    tf.summary.scalar('coefficients/' + name, coeff, step=self.step)
                    if self.latest_grads:
                        tf.summary.scalar('gradients/' + name, self.latest_grads[idx], step=self.step)

                # Histogram data
                if self.latest_grads:
                    tf.summary.histogram('gradients', self.latest_grads, step=self.step)
                tf.summary.histogram('coefficients', bmfc.trainables(self.fit_coeffs), step=self.step)

                tf.summary.flush()

        # If the script context has `log` set to True AND we had `signal_coeffs` passed on creation,
        #  then log the true signal coefficient values for this step
        if self.signal_writer:
            with self.signal_writer.as_default():
                for coeff in bmfc.trainables(self.fit_coeffs):
                    idx = self.fit_coeffs.index(coeff)
                    tf.summary.scalar('coefficients/' + bmfc.names[idx], self.signal_coeffs[idx], step=self.step)

                tf.summary.flush()

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
            "nll:", self.latest_nll,
            "grad_max:", self.latest_grad_max,
            "grad_mean:", self.latest_grad_mean,
            "grad_total:", self.latest_grad_total,
        )
        self.script.stdout("fit:   ", bmfc.to_str(self.fit_coeffs))
        if self.signal_coeffs:
            self.script.stdout("signal:", bmfc.to_str(self.signal_coeffs))
