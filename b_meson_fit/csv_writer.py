import csv
import os

import b_meson_fit.coeffs as bmfc


class CsvWriter:
    current_id = 0
    handle = None

    def __init__(self, filename):
        written_headers = False
        if os.path.isfile(filename):
            with open(filename, 'r', newline='') as previous_csv:
                rows = csv.DictReader(previous_csv)
                written_headers = True
                for row in rows:
                    self.current_id = int(row['id'])

        self.handle = open(filename, "a", newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=(['id'] + bmfc.names))
        if not written_headers:
            self.writer.writeheader()

    def __del__(self):
        if self.handle:
            self.handle.close()

    def write_coeffs(self, coeffs):
        self.current_id = self.current_id + 1
        coeff_floats = [c.numpy() for c in coeffs]
        row = {'id': self.current_id, **dict(zip(bmfc.names, coeff_floats))}
        self.writer.writerow(row)
        self.handle.flush()

