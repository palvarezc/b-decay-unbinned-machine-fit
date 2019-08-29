#!/bin/bash

set -e

./bin/plot_amplitudes.py --signal-model SM --write-svg 'results/amplitude-SM-%name%.svg'
./bin/plot_angular_observables.py --signal-model SM --write-svg 'results/angular_observable-SM-%name%.svg'
./bin/plot_breit_wigner.py --write-svg results/breit_wigner.svg
./bin/plot_frac_s.py --signal-model SM --write-svg results/frac_s-SM.svg
./bin/plot_differential_decay_rate.py --signal-model SM --write-svg results/differential_decay_rate-SM.svg
./bin/plot_signal.py --signal-model SM --write-svg 'results/signal-SM-%name%.svg'
./bin/plot_confidence.py --write-svg 'results/confidence-SM-%name%.svg' results/SM_rate-0.05_b1-def_b2-def_eps-def_fi-def.csv
./bin/plot_q_test_statistic.py --write-svg 'results/q_test_statistic.svg' results/q_test_stat-SM.txt results/q_test_stat-NP.txt