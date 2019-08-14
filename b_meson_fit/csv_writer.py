"""Write coefficients to a CSV file for ensamble runs.

File will contain columns id,normalized_nll,a_para_l_re_alpha,...,a_zero_r_im_gamma

Writer will append to a current file if it exists and the signal coefficients match.

ID is automatic and will be incremented for each row. Signal is written as ID 0.
"""
import csv
import os

import b_meson_fit.coeffs as bmfc


class CsvWriter:
    current_id = 0
    handle = None

    def __init__(self, file_path, signal_coeffs):
        """Args:
            file_path (str): Full path of CSV to write
            signal_coeffs (list of tensors): Signal coefficients used for this ensemble
        """
        written_headers = None
        written_signal = None

        # If we already have a file then assume headers are written and get latest ID
        if os.path.isfile(file_path):
            with open(file_path, 'r', newline='') as previous_csv:
                rows = csv.DictReader(previous_csv)
                written_headers = rows.fieldnames
                if written_headers:
                    try:
                        written_signal = next(rows)
                    except StopIteration:
                        pass

                    if written_signal is not None:
                        expected_signal = self._signal_row(signal_coeffs)
                        if int(written_signal['id']) != 0:
                            raise RuntimeError('Signal row 0 not found in file {}'.format(file_path))
                        # written_signal has all fields as strings so cast all expected_signal fields to the same
                        #  before comparison
                        if dict(written_signal) != {k: str(v) for k, v in expected_signal.items()}:
                            raise RuntimeError('Current signal coeffs do not match with file {}'.format(file_path))

                        for row in rows:
                            self.current_id = int(row['id'])

        # Open the file for writing and write headers if we haven't written before
        self.handle = open(file_path, "a", newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=(['id', 'normalized_nll'] + bmfc.names + ['time_taken']))

        if written_headers is None:
            self.writer.writeheader()
        if written_signal is None:
            self._write_signal(signal_coeffs)

    def __del__(self):
        """Cleanup on close"""
        if self.handle:
            self.handle.close()

    def write_coeffs(self, normalized_nll, coeffs, time_taken):
        """Write row of coefficients

        Args:
            normalized_nll (float): Normalized NLL to write
            coeffs (list of tensors): Coefficient list to write
            time_taken (float): Seconds this fit took
        """
        self.current_id = self.current_id + 1
        self.writer.writerow(self._row(normalized_nll, coeffs, time_taken))
        self.handle.flush()

    def _write_signal(self, signal_coeffs):
        self.writer.writerow(self._signal_row(signal_coeffs))
        self.handle.flush()

    def _signal_row(self, signal_coeffs):
        return self._row(0.0, signal_coeffs, 0.0)

    def _row(self, normalized_nll, coeffs, time_taken):
        coeff_floats = [c.numpy() for c in coeffs]
        return {
            'id': self.current_id,
            'normalized_nll': normalized_nll,
            **dict(zip(bmfc.names, coeff_floats)),
            'time_taken': time_taken
        }

