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

tf.enable_v2_behavior()


class Script:
    """Script context manager class
    """
    device = None

    def __init__(self, device='GPU:0'):
        self.name = os.path.basename(os.path.splitext(sys.argv[0])[0])

        if device:
            self.device = tf.device('/device:' + device)

    def __enter__(self):
        """Print on script startup"""
        stdout('Starting {}'.format(self.name))
        self.start_time = time.time()

        if self.device:
            self.device.__enter__()

        return self

    def __exit__(self, *ex_info):
        """Print goodbye and timing on script shutdown"""
        if self.device:
            self.device.__exit__(*ex_info)

        time_elapsed = time.time() - self.start_time
        stdout('')
        stdout('Finished {0} in {1:0.1f}s'.format(self.name, time_elapsed))


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
