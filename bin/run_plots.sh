#!/bin/bash

set -e

log_file="results/run_plots.log"
> ${log_file}
exec 1> >(tee ${log_file})
exec 2>&1

./bin/plot_breit_wigner.py --write-svg results/breit_wigner.svg
for model in SM NP
do
    ./bin/plot_amplitudes.py \
        --signal-model ${model} \
        --write-svg "results/amplitude-${model}-%name%.svg"
    ./bin/plot_angular_observables.py \
        --signal-model ${model} \
        --write-svg "results/angular_observable-${model}-%name%.svg"
    ./bin/plot_frac_s.py \
        --signal-model ${model} \
        --write-svg results/frac_s-${model}.svg
    ./bin/plot_differential_decay_rate.py \
        --signal-model ${model} \
        --write-svg results/differential_decay_rate-${model}.svg
    ./bin/plot_signal.py \
        --signal-model ${model} \
        --write-svg "results/signal-${model}-%name%.svg"
done

for csv in $(find results/ -name 'fit-*.csv')
do
    info=${csv:12:-4}
    ./bin/plot_confidence.py \
        --write-svg "results/confidence-${info}-%name%.svg" \
        ${csv}
done

for signal_count in 600 2400
do
    ./bin/plot_q_test_statistic.py \
        --write-svg "results/q_test_statistic-${signal_count}.svg" \
        results/q_test_stat-${signal_count}-SM.txt \
        results/q_test_stat-${signal_count}-NP.txt
done