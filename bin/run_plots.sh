#!/bin/bash

set -e
./bin/plot_amplitudes.py --signal-model SM --write-svg 'results/amplitude-SM-%name%.svg'
./bin/plot_signal.py --signal-model SM --write-svg 'results/signal-SM-%name%.svg'
./bin/plot_q_test_statistic.py --write-svg 'results/q_test_statistic.svg' results/q_test_stat-SM.txt results/q_test_stat-NP.txt