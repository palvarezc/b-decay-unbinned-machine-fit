import io
import os
import tempfile
import tensorflow.compat.v2 as tf
import unittest

import b_meson_fit.coeffs as bmfc
import b_meson_fit.csv_writer as bmfw

tf.enable_v2_behavior()


class TestCsv(unittest.TestCase):

    def test_write_header_only(self):
        """Check that file header writing works correctly"""
        tmp_file = tempfile.mktemp()
        self.maxDiff = None

        bmfw.CsvWriter(tmp_file)
        self._compare('csv_writer_headers_only.csv', tmp_file, 'Non-existent file gets headers')
        bmfw.CsvWriter(tmp_file)
        self._compare('csv_writer_headers_only.csv', tmp_file, 'Existing file does not get duplicate headers')

    def test_write_rows(self):
        """Check writing rows to new and existing files works as expected"""
        tmp_file = tempfile.mktemp()
        self.maxDiff = None

        # Ensure fit coefficients are constants
        bmfc.fit_default = 12.345

        csv_writer = bmfw.CsvWriter(tmp_file)
        csv_writer.write_coeffs(tf.constant(1.2), bmfc.signal(bmfc.SM), 14.8)
        csv_writer.write_coeffs(tf.constant(3.4), bmfc.fit(), 15.3)
        csv_writer.write_coeffs(tf.constant(5.6), bmfc.signal(bmfc.NP), 13.9)
        self._compare('csv_writer_rows_first_write.csv', tmp_file, 'Non-existent file gets rows written correctly')

        csv_writer = bmfw.CsvWriter(tmp_file)
        csv_writer.write_coeffs(tf.constant(7.8), bmfc.fit(), 14.0)
        self._compare('csv_writer_rows_append.csv', tmp_file, 'Existing file gets rows appended correctly')

    def _compare(self, expected_filename, actual_filepath, msg):
        self.assertListEqual(
            self._file_contents(self._test_data_path(expected_filename)),
            self._file_contents(actual_filepath),
            msg
        )

    @staticmethod
    def _file_contents(filepath):
        handle = io.open(filepath)
        contents = list(handle)
        handle.close()
        return contents

    @staticmethod
    def _test_data_path(filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data', filename)


