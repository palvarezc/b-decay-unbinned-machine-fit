"""
Handle Tensorboard log file paths

Todo:
    * Make this OS portable with file paths
"""
import datetime
import os
import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()

_now = datetime.datetime.now()


class Log:
    top_dir = 'logs'
    """Attributes:
        top_dir (str): Project-level folder to write logs to
    """

    def __init__(self, name):
        """Args:
            name (str): Subfolder under `top_dir` to write logs to
        """
        self.name = name
        self.date = _now.strftime("%Y%m%d-%H%M%S")
        self._suffix = ''

    @property
    def dir(self):
        """str: Return path to logs folder

        Path format from project directory is `top_dir`/`name`/`date`[/`suffix`]
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        return "{}/../{}/{}{}".format(
            current_dir,
            self.top_dir,
            self.prefix,
            "/{}".format(self.suffix) if self.suffix else ""
        )

    @property
    def prefix(self):
        """str: Get the `name`/`date` prefix for these logs.

        Useful to use in Tensorboard as a regex filter
        """
        return "{}/{}".format(self.name, self.date)

    @property
    def suffix(self):
        """str: Optional additional subfolder for logging"""
        return self._suffix

    @suffix.setter
    def suffix(self, s):
        self._suffix = s

    def writer(self, one_off_suffix=None):
        """Get a new log summary FileWriter

        Args:
            one_off_suffix: Optional suffix override for just this writer.

        Returns:
            FileWriter
        """
        old_suffix = None
        if one_off_suffix:
            old_suffix = self.suffix
            self.suffix = one_off_suffix
        writer = tf.summary.create_file_writer(logdir=self.dir)
        if old_suffix:
            self.suffix = old_suffix
        return writer
