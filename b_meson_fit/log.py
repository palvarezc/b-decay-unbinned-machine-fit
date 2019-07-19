"""
Handle Tensorboard logging

Note:
    Using this slows iterations down significantly and should only be used for development.
"""
import datetime
import os
import tensorflow.compat.v2 as tf

import b_meson_fit.coeffs as bmfc
from b_meson_fit.script import stdout

tf.enable_v2_behavior()

_now = datetime.datetime.now()


class Log:
    top_dir = 'logs'
    signal_name = 'signal'
    """Attributes:
        top_dir (str): Project-level folder to write logs to
        signal_name (str): Suffix for signal coefficient plots
    """

    def __init__(self, name):
        """Args:
            name (str): Subfolder under `top_dir` to write logs to
        """
        self.name = name
        self.date = _now.strftime("%Y%m%d-%H%M%S")
        self.writers = {}
        self._suffix = ''

        stdout('')
        stdout(
            'Start Tensorboard from the project folder with ' +
            '`tensorboard --logdir={}/ --host=127.0.0.1 --port=6006\''.format(self.top_dir) +
            ' and navigate to http://127.0.0.1:6006'
        )
        stdout('Filter regex: {}'.format(self._prefix()))
        stdout('')

    def dir(self, suffix=None):
        """Return absolute path to logs folder.

        From project directory this is ./`top_dir`/`name`/`date`[/`suffix`]

        Args:
            suffix (str, optional): Optional suffix directory

        Return:
            str: Logs path
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        path_parts = [current_dir, '..', self.top_dir, self._prefix()]
        if suffix:
            path_parts.append(suffix)
        return os.path.join(*path_parts)

    def signal_line(self, fit_coeffs, signal_coeffs, iterations):
        """Write constant signal coefficient values for a number of iterations. Will only write signal values for
        trainable fit coefficients, so if you only want to train two coefficients, you will only get two plots.

        Useful for scripts that do different runs for a fixed number of iterations comparing hyper-parameters.
        Scripts that do not use a fixed number of iterations should use coefficients() instead.

        Args:
            fit_coeffs (list of tensors): Fit coefficients that will be used for this run. Used to work out
                which coefficients will be trainable.
            signal_coeffs (list of tensors): Signal coefficients to write.
            iterations (int): Number of iterations to write.
        """
        with self._writer(self.signal_name).as_default():
            for coeff in bmfc.trainables(fit_coeffs):
                idx = fit_coeffs.index(coeff)
                for i in range(iterations):
                    tf.summary.scalar('coefficients/' + bmfc.names[idx], signal_coeffs[idx], step=i)

            tf.summary.flush()

    def coefficients(self, coeffs_name, optimizer, signal_coeffs=None):
        """Write fit coefficients, gradients and optionally signal coefficients for this optimization step

           Args:
               coeffs_name (str): Name of these fit coefficients.
               optimizer (Optimizer): Optimizer class in use,
               signal_coeffs (list of tensors): Optional signal coefficients if they are also to be written.
           """
        # Handle fit coefficients and gradients
        with self._writer(coeffs_name).as_default():
            tf.summary.scalar('normalized_nll', optimizer.normalized_nll, step=optimizer.step)
            if optimizer.grads:
                tf.summary.scalar('gradients/max', optimizer.grad_max, step=optimizer.step)
                tf.summary.scalar('norms/global', tf.linalg.global_norm(optimizer.grads), step=optimizer.step)

            # All trainable coefficients and gradients as individual scalars
            for idx, coeff in enumerate(optimizer.trainables):
                name = bmfc.names[optimizer.fit_coeffs.index(coeff)]
                tf.summary.scalar('coefficients/' + name, coeff, step=optimizer.step)
                if optimizer.grads:
                    tf.summary.scalar('gradients/' + name, optimizer.grads[idx], step=optimizer.step)
                    tf.summary.scalar('norms/' + name, tf.norm(optimizer.grads[idx]), step=optimizer.step)

        # Optionally handle signal coefficients
        if signal_coeffs:
            with self._writer(self.signal_name).as_default():
                for coeff in optimizer.trainables:
                    idx = optimizer.fit_coeffs.index(coeff)
                    tf.summary.scalar('coefficients/' + bmfc.names[idx], signal_coeffs[idx], step=optimizer.step)

    def _prefix(self):
        return os.path.join(self.name, self.date)

    def _writer(self, suffix):
        if suffix not in self.writers:
            self.writers[suffix] = tf.summary.create_file_writer(logdir=self.dir(suffix))

        return self.writers[suffix]
