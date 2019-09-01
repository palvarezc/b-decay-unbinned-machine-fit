"""Write coefficients to a CSV file for ensemble runs. ID is automatic and will be incremented for each row.

Two classes are available:

FitWriter:  Write results of a fit. Also writes the signal coeffs as id == 0.
            Will refuse to continue if the signal coefficients have changed.
            Will contain columns id,normalized_nll,a_para_l_re_alpha,...,a_zero_r_im_gamma

QWriter:    Write results of a Q statistic run.
            Will contain columns id,test_a_00_l_re_alpha,...,test_nll,null_a_00_l_re_alpha,null_nll,...,q,time_taken
"""
import csv
import os

import b_meson_fit.coeffs as bmfc


class CsvWriter:
    """Parent class"""
    current_id = 0
    handle = None

    def __init__(self, file_path, headers, extra_header_check=None, extra_header_func=None):
        """Args:
            file_path (str): Full path of CSV to write
            headers (list of str): Names of headers to write. Should exclude ID
            extra_header_check (func, optional): Optional function to run after the headers have been read.
                Should take 1 argument of CSV rows from csv.DictReader. Should return True or False indicating
                whether extra_header_func needs running
            extra_header_func (func, optional): This function is run after headers are written if extra_header_check
                returns True. Should take no arguments
        """
        written_headers = None
        run_extra_header_func = False

        # If we already have a file then assume headers are written and get latest ID
        if os.path.isfile(file_path):
            with open(file_path, 'r', newline='') as previous_csv:
                rows = csv.DictReader(previous_csv)
                written_headers = rows.fieldnames
                if written_headers:
                    if extra_header_check is not None:
                        run_extra_header_func = extra_header_check(rows)
                        # Reset file back to start
                        previous_csv.seek(0)
                        rows = csv.DictReader(previous_csv)

                    for row in rows:
                        self.current_id = int(row['id'])

        # Open the file for writing and write headers if we haven't written before
        self.handle = open(file_path, "a", newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=(['id'] + headers))

        if written_headers is None:
            self.writer.writeheader()
        if extra_header_func is not None and run_extra_header_func:
            extra_header_func()

    def __del__(self):
        """Cleanup on close"""
        if self.handle:
            self.handle.close()

    def write(self, row, increment_id=True):
        if increment_id:
            self.current_id = self.current_id + 1
        self.writer.writerow({'id': self.current_id, **row})
        self.handle.flush()


class FitWriter(CsvWriter):
    def __init__(self, file_path, signal_coeffs):
        def check_signal(rows):
            """Check if the signal has been previously written and that it matches what we expect"""
            written_signal = None
            try:
                written_signal = next(rows)
                # print(written_signal)
            except StopIteration:
                pass

            if written_signal is not None:
                expected_signal = {'id': 0, **self._signal_row(signal_coeffs)}
                if int(written_signal['id']) != 0:
                    raise RuntimeError('Signal row 0 not found in file {}'.format(file_path))
                # written_signal has all fields as strings so cast all expected_signal fields to the same
                #  before comparison
                if dict(written_signal) != {k: str(v) for k, v in expected_signal.items()}:
                    raise RuntimeError('Current signal coeffs do not match with file {}'.format(file_path))

            return written_signal is None

        super().__init__(
            file_path,
            (['normalized_nll'] + bmfc.names + ['time_taken']),
            check_signal,
            lambda: self.write(self._signal_row(signal_coeffs), False) # Write signal if absent
        )

    def write_coeffs(self, normalized_nll, coeffs, time_taken):
        """Write row of coefficients

        Args:
            normalized_nll (float): Normalized NLL to write
            coeffs (list of tensors): Coefficient list to write
            time_taken (float): Seconds this fit took
        """
        self.write(self._row(normalized_nll, coeffs, time_taken))

    def _signal_row(self, signal_coeffs):
        return self._row(0.0, signal_coeffs, 0.0)

    @staticmethod
    def _row(normalized_nll, coeffs, time_taken):
        coeff_floats = [c.numpy() for c in coeffs]
        return {
            'normalized_nll': normalized_nll,
            **dict(zip(bmfc.names, coeff_floats)),
            'time_taken': time_taken
        }


class QWriter(CsvWriter):
    def __init__(self, file_path):
        headers = []
        for model in ['test', 'null']:
            for idx in bmfc.s_wave_idxs:
                headers.append('{}_{}'.format(model, bmfc.names[idx]))
            headers.append('{}_nll'.format(model))
        headers.append('q')
        headers.append('time_taken')

        super().__init__(file_path, headers)

    def write_q(self, test_coeffs, test_nll, null_coeffs, null_nll, q, time_taken):
        """Write row of coefficients

        Args:
            test_coeffs (list of tensors): Coefficient list for test hypothesis
            test_nll (float): NLL for test hypothesis
            null_coeffs (list of tensors): Coefficient list for null hypothesis
            null_nll (float): NLL for null hypothesis
            q (float): Value of Q test statistic
            time_taken (float): Seconds this fit took
        """
        row = {}
        for idx in bmfc.s_wave_idxs:
            row['test_{}'.format(bmfc.names[idx])] = test_coeffs[idx].numpy()
        row['test_nll'] = test_nll
        for idx in bmfc.s_wave_idxs:
            row['null_{}'.format(bmfc.names[idx])] = null_coeffs[idx].numpy()
        row['null_nll'] = null_nll
        row['q'] = q
        row['time_taken'] = time_taken

        self.write(row)
