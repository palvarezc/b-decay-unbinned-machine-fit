"""Context manager for use in bin/* scripts

Example:
    import b_meson_fit as bmf

    params={'param1': 'something', 'param2': [1, 2, 3], 'param3': ['a', 'b', 'c']}
    with bmf.Script(params=params) as script:
        do_script_stuff();
        script.stdout('I just did stuff')
"""
import os
import sys
import tensorflow.compat.v2 as tf
import time

import b_meson_fit.log as bmfl

tf.enable_v2_behavior()


class Script:
    """Script context manager class
    """

    def __init__(self, params=None, log=False):
        """Args:
            params (dict of str: any): Parameter names and values script is being used with. Values can be iterable.
            log (bool): Whether we're writing logs for Tensorboard.
        """
        self.name = os.path.basename(os.path.splitext(sys.argv[0])[0])
        self.params = params
        self.log = bmfl.Log(self.name) if log else None

    def __enter__(self):
        """Print hello and `params` (if set) on script startup.

        Also prints how to start and access Tensorboard is `log` is True
        """
        self.stdout('Starting {}'.format(self.name), end='')
        if self.params:
            self.stdout(' with settings:')
            for n, v in self.params.items():
                try:
                    _ = iter(v)
                except TypeError:
                    v_str = str(v)
                else:
                    v_str = ', '.join(map(str, iter(v)))
                self.stdout(' * {}: {}'.format(n, v_str))
        else:
            self.stdout('')
        self.stdout('')

        if self.log:
            self.stdout(
                'Start Tensorboard from the project folder with ' +
                '`tensorboard --logdir={}/ --host=127.0.0.1 --port=6006\''.format(self.log.top_dir) +
                ' and navigate to http://127.0.0.1:6006'
            )
            self.stdout('Filter regex: {}'.format(self.log.prefix))
            self.stdout('')

        self.start_time = time.time()

        return self

    def __exit__(self, _type, _value, _traceback):
        """Print goodbye on script shutdown"""
        time_elapsed = time.time() - self.start_time
        self.stdout('')
        self.stdout('Finished {0} in {1:0.1f}s'.format(self.name, time_elapsed))

    @staticmethod
    def stdout(*args, **kwargs):
        """Print to stdout"""
        tf.print(*args, output_stream=sys.stdout, **kwargs)

    @staticmethod
    def stderr(*args, **kwargs):
        """Print to stderr"""
        tf.print(*args, output_stream=sys.stderr, **kwargs)

    @staticmethod
    def user_is_root():
        """Check if script is running as effective root"""
        return os.geteuid() == 0
