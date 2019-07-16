import io
import os
import tempfile
import unittest

import b_meson_fit.coeffs as bmfc
import b_meson_fit.csv_writer as bmfw

# Ensure fit coefficients are constants
bmfc.fit_default = 12.345


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

        csv_writer = bmfw.CsvWriter(tmp_file)
        csv_writer.write_coeffs(bmfc.signal())
        csv_writer.write_coeffs(bmfc.fit())
        csv_writer.write_coeffs(bmfc.signal())
        self._compare('csv_writer_rows_first_write.csv', tmp_file, 'Non-existent file gets rows written correctly')

        csv_writer = bmfw.CsvWriter(tmp_file)
        csv_writer.write_coeffs(bmfc.fit())
        self._compare('csv_writer_rows_append.csv', tmp_file, 'Existing file gets rows appended correctly')

    def _compare(self, expected_filename, actual_filepath, msg):
        expected_handle = io.open(self._test_data_path(expected_filename))
        expected = list(expected_handle)
        expected_handle.close()

        actual_handle = io.open(actual_filepath)
        actual = list(actual_handle)
        actual_handle.close()

        self.assertListEqual(expected, actual, msg)

    @staticmethod
    def _test_data_path(filename):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_data', filename)


