"""Write coefficients to a CSV file for ensamble runs.

File will contain columns id,normalized_nll,a_para_l_re_alpha,...,a_zero_r_im_gamma

Writer will append to a current file if it exists.

ID is automatic and will be incremented for each row.
"""
import csv
import os

import b_meson_fit.coeffs as bmfc


class CsvWriter:
    current_id = 0
    handle = None

    def __init__(self, file_path):
        """Args:
            file_path (str): Full path of CSV to write
        """
        written_headers = False

        # If we already have a file then assume headers are written and get latest ID
        if os.path.isfile(file_path):
            with open(file_path, 'r', newline='') as previous_csv:
                rows = csv.DictReader(previous_csv)
                written_headers = True
                for row in rows:
                    self.current_id = int(row['id'])

        # Open the file for writing and write headers if we haven't written before
        self.handle = open(file_path, "a", newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=(['id', 'normalized_nll'] + bmfc.names))
        if not written_headers:
            self.writer.writeheader()

    def __del__(self):
        if self.handle:
            self.handle.close()

    def write_coeffs(self, normalized_nll, coeffs):
        """Write row of coefficients

        Args:
            coeffs (list of tensors): Coefficient list to write
        """
        self.current_id = self.current_id + 1
        coeff_floats = [c.numpy() for c in coeffs]
        row = {'id': self.current_id, 'normalized_nll': normalized_nll.numpy(), **dict(zip(bmfc.names, coeff_floats))}
        self.writer.writerow(row)
        self.handle.flush()

