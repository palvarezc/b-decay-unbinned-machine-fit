import datetime
import os
import unittest

from b_meson_fit import log


class TestLog(unittest.TestCase):

    def setUp(self):
        # 2010-11-12 13:14:15
        log._now = datetime.datetime(2010, 11, 12, 13, 14, 15)

    def test_log_paths(self):
        """Check Log returns expected paths"""
        log_inst = log.Log('unittest')
        project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

        self.assertEqual(
            os.path.realpath(os.path.join(project_dir, 'logs', 'unittest', '20101112-131415')),
            os.path.realpath(log_inst.dir())
        )

        self.assertEqual(
            os.path.realpath(os.path.join(project_dir, 'logs', 'unittest', '20101112-131415', 'somesuffix')),
            os.path.realpath(log_inst.dir('somesuffix'))
        )


if __name__ == '__main__':
    unittest.main()
