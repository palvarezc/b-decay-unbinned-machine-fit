import io
import os
import shutil
import tempfile
import tensorflow.compat.v2 as tf
import unittest

import b_meson_fit.coeffs as bmfc
import b_meson_fit.csv_writer as bmfw

tf.enable_v2_behavior()


class TestCsv(unittest.TestCase):

    def test_write_headers_and_signal(self):
        """Check that file header and signal writing works correctly"""
        tmp_file = tempfile.mktemp()
        self.maxDiff = None

        signal_coeffs = bmfc.signal(bmfc.SM)

        bmfw.CsvWriter(tmp_file, signal_coeffs)
        self._compare('csv_writer_headers_and_signal.csv', tmp_file, 'Non-existent file gets headers and signal')
        bmfw.CsvWriter(tmp_file, signal_coeffs)
        self._compare(
            'csv_writer_headers_and_signal.csv',
            tmp_file,
            'Existing file does not get duplicate headers or signal'
        )

    def test_write_rows(self):
        """Check writing rows to new and existing files works as expected"""
        tmp_file = tempfile.mktemp()
        self.maxDiff = None

        signal_coeffs = bmfc.signal(bmfc.SM)

        # Ensure fit coefficients are constants
        bmfc.fit_default = 12.345

        csv_writer = bmfw.CsvWriter(tmp_file, signal_coeffs)
        csv_writer.write_coeffs(1.2, bmfc.signal(bmfc.SM), 14.8)
        csv_writer.write_coeffs(3.4, bmfc.fit(), 15.3)
        csv_writer.write_coeffs(5.6, bmfc.signal(bmfc.NP), 13.9)
        self._compare('csv_writer_rows_first_write.csv', tmp_file, 'Non-existent file gets rows written correctly')

        csv_writer = bmfw.CsvWriter(tmp_file, signal_coeffs)
        csv_writer.write_coeffs(7.8, bmfc.fit(), 14.0)
        self._compare('csv_writer_rows_append.csv', tmp_file, 'Existing file gets rows appended correctly')

    def test_write_headers_only(self):
        """Check that opening a file with only headers works correctly. In normal operation this should not happen"""
        tmp_file = tempfile.mktemp()
        self.maxDiff = None

        signal_coeffs = bmfc.signal(bmfc.SM)

        shutil.copyfile(self._test_data_path('csv_writer_headers_only.csv'), tmp_file)
        bmfw.CsvWriter(tmp_file, signal_coeffs)
        self._compare(
            'csv_writer_headers_and_signal.csv',
            tmp_file,
            'Existing file with only headers gets signal written'
        )

    def test_exception_raised_if_signal_does_not_match(self):
        """Check that signal coefficients must match when opening an existing file"""
        tmp_file = tempfile.mktemp()
        bmfw.CsvWriter(tmp_file, bmfc.signal(bmfc.SM))
        with self.assertRaises(RuntimeError):
            bmfw.CsvWriter(tmp_file, bmfc.signal(bmfc.NP))

    def _compare(self, expected_filename, actual_filepath, msg):
        # shutil.copyfile(actual_filepath, self._test_data_path(expected_filename))
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


if __name__ == '__main__':
    unittest.main()
