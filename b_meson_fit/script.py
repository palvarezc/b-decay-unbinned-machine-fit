"""Context manager for use in bin/* scripts

Example:
    import b_meson_fit as bmf

    with bmf.Script(params=params) as script:
        do_script_stuff();
        bmf.stdout('My name is {} and I did stuff'.format(script.name))
"""
import os
import sys
import tensorflow.compat.v2 as tf
import time

# Force deprecation warnings off to stop them breaking our progress bar(s).
# The warnings with TF 1.14 are from TF internal code anyway.
# You should probably comment out if whilst upgrading Tensorflow.
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.enable_v2_behavior()


class Script:
    """Script context manager class
    """
    device_default = 'GPU:0'

    def __init__(self, device=device_default):
        self.name = os.path.basename(os.path.splitext(sys.argv[0])[0])

        self.device = device
        if self.device:
            self._device_ctx = tf.device('/device:' + device)

        self._start_times = {}

    def __enter__(self):
        """Print on script startup"""
        stdout('Starting {}{}'.format(self.name, ' on device {}'.format(self.device) if self.device else ''))
        stdout('')
        self.timer_start('script')

        if self.device:
            self._device_ctx.__enter__()

        return self

    def __exit__(self, *ex_info):
        """Print goodbye and timing on script shutdown"""
        if self.device:
            self._device_ctx.__exit__(*ex_info)

        stdout('')
        stdout('Finished {0} in {1:0.1f}s'.format(self.name, self.timer_elapsed('script')))

    def timer_start(self, name):
        """Start a timer"""
        self._start_times[name] = time.time()

    def timer_elapsed(self, name):
        """Get the elapsed timer time"""
        if not self._start_times[name]:
            raise RuntimeError('{} timer has not been started'.format(name))
        return time.time() - self._start_times[name]


def stdout(*args, **kwargs):
    """Print to stdout"""
    with tf.device('/device:CPU:0'):
        tf.print(*args, output_stream=sys.stdout, **kwargs)


def stderr(*args, **kwargs):
    """Print to stderr"""
    with tf.device('/device:CPU:0'):
        tf.print(*args, output_stream=sys.stderr, **kwargs)


def user_is_root():
    """Check if script is running as effective root"""
    return os.geteuid() == 0
