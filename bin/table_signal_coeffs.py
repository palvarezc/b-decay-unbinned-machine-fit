#!/usr/bin/env python
"""Script to output signal coefficients as LaTeX table for publication"""

import os
# Disable non-important TF log lines
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import b_meson_fit as bmf

signal_coeffs = {}
for model in bmf.coeffs.signal_models:
    coeffs = bmf.coeffs.signal(model)
    for c_idx, coeff in enumerate(coeffs):
        c_name = bmf.coeffs.names[c_idx]
        if c_name not in signal_coeffs:
            signal_coeffs[c_name] = {}

        signal_coeffs[c_name][model] = "{:+.5f}".format(coeff.numpy())


print('\\begin{table}[h!]')
print('\\centering')
print(' \\begin{{tabular}}{{|{}|}}'.format('|'.join(['c'] * (len(bmf.coeffs.signal_models) * 3 + 1))))
print(' \\hline')
print(
    ' \\multirow{{3}}{{*}}{{Component}} & \\multicolumn{{{}}}{{c|}}{{Model}} \\\\'.format(
        len(bmf.coeffs.signal_models) * 3
    )
)
print(' \\cline{{2-{}}}'.format(len(bmf.coeffs.signal_models) * 3 + 1))
row = []
for model in bmf.coeffs.signal_models:
    row.append('\\multicolumn{{3}}{{c|}}{{{}}}'.format(model))
print(' & {} \\\\'.format(' & '.join(row)))
print(' \\cline{{2-{}}}'.format(len(bmf.coeffs.signal_models) * 3 + 1))
print(' & {} \\\\'.format(' & '.join(bmf.coeffs.param_latex_names * len(bmf.coeffs.signal_models))))
print(' \\hline')

for a_idx, amplitude_latex_name in enumerate(bmf.coeffs.amplitude_latex_names):
    row = [amplitude_latex_name]
    for model in bmf.coeffs.signal_models:
        for p_idx in range(bmf.coeffs.param_count):
            c_idx = 3 * a_idx + p_idx
            c_name = bmf.coeffs.names[c_idx]
            row.append(signal_coeffs[c_name][model])

    print(' ', end='')
    print(' & '.join(row), end=' \\\\ \n \\hline\n')

print(' \\end{tabular}')
print('\\caption{Write me}')
print('\\label{table:signal-coeffs}')
print('\\end{table}')
