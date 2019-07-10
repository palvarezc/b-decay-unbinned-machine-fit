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

        self.assertEqual('unittest/20101112-131415', log_inst.prefix)
        self.assertEqual(
            os.path.realpath('../logs/unittest/20101112-131415'),
            os.path.realpath(log_inst.dir)
        )

        log_inst.suffix = 'somesuffix'
        self.assertEqual('somesuffix', log_inst.suffix)
        self.assertEqual(
            os.path.realpath('../logs/unittest/20101112-131415/somesuffix'),
            os.path.realpath(log_inst.dir)
        )
