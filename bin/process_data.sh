#!/bin/bash

set -o errexit

# Disable info logging
export TF_CPP_MIN_LOG_LEVEL=1

# Go to project directory
cd ${BASH_SOURCE[0]%/*}/..

run() {
    echo "**************"
    echo "$@"
    "$@"
    echo "**************"
    echo
}

log_file="results/process_data.log"
> ${log_file}
exec 1> >(tee ${log_file})
exec 2>&1

run ./bin/plot_breit_wigner.py --write-svg results/breit_wigner.svg
for model in SM NP
do
    run ./bin/plot_amplitudes.py \
        --signal-model ${model} \
        --write-svg "results/amplitude-${model}-%name%.svg"
    run ./bin/plot_angular_observables.py \
        --signal-model ${model} \
        --write-svg "results/angular_observable-${model}-%name%.svg"
    run ./bin/plot_frac_s.py \
        --signal-model ${model} \
        --write-svg results/frac_s-${model}.svg
    run ./bin/plot_differential_decay_rate.py \
        --signal-model ${model} \
        --write-svg results/differential_decay_rate-${model}.svg
    run ./bin/plot_signal.py \
        --signal-model ${model} \
        --write-svg "results/signal-${model}-%name%.svg"
done

for model in SM NP
do
    # Rates
    plots=""
    for csv in $(find results/ -name "fit-${model}_rate-*_b1-def_b2-def_eps-def_fi-def.csv" -exec realpath --relative-to=. {} \; | sort -n)
    do
        rate=$(echo "${csv}" | cut -d '_' -f 2 | cut -d '-' -f 2)
        plots="${plots} ${csv}:${rate}"
    done
    run ./bin/plot_fit_distributions.py \
        --write-svg "results/fit-${model}-rates-%name%.svg" \
        ${plots}
    run ./bin/plot_pulls.py \
        --write-svg "results/pull-${model}-rates-%name%.svg" \
        ${plots}

    # Beta1s
    plots=""
    for csv in $(find results/ -name "fit-${model}_rate-def_b1-*_b2-def_eps-def_fi-def.csv" -exec realpath --relative-to=. {} \; | sort -n)
    do
        beta=$(echo "${csv}" | cut -d '_' -f 3 | cut -d '-' -f 2)
        plots="${plots} ${csv}:${beta}"
    done
    run ./bin/plot_fit_distributions.py \
        --write-svg "results/fit-${model}-beta1-%name%.svg" \
        ${plots}
    run ./bin/plot_pulls.py \
        --write-svg "results/pull-${model}-beta1-%name%.svg" \
        ${plots}

    # Beta2s
    plots=""
    for csv in $(find results/ -name "fit-${model}_rate-def_b1-def_b2-*_eps-def_fi-def.csv" -exec realpath --relative-to=. {} \; | sort -n)
    do
        beta=$(echo "${csv}" | cut -d '_' -f 4 | cut -d '-' -f 2)
        plots="${plots} ${csv}:${beta}"
    done
    run ./bin/plot_fit_distributions.py \
        --write-svg "results/fit-${model}-beta2-%name%.svg" \
        ${plots}
    run ./bin/plot_pulls.py \
        --write-svg "results/pull-${model}-beta2-%name%.svg" \
        ${plots}

    # Epsilons
    plots=""
    for csv in $(find results/ -name "fit-${model}_rate-def_b1-def_b2-def_eps-*_fi-def.csv" -exec realpath --relative-to=. {} \; | sort -n)
    do
        eps=$(echo "${csv}" | cut -d '_' -f 5 | awk -F'eps-' '{print $2}')
        plots="${plots} ${csv}:${eps}"
    done
    run ./bin/plot_fit_distributions.py \
        --write-svg "results/fit-${model}-eps-%name%.svg" \
        ${plots}
    run ./bin/plot_pulls.py \
        --write-svg "results/pull-${model}-eps-%name%.svg" \
        ${plots}

    # Show discrete symmetries
    run ./bin/plot_fit_distributions.py \
        --write-svg "results/fit-${model}-TWICE_CURRENT_SIGNAL_ANY_SIGN-%name%.svg" \
        "results/fit-${model}_rate-def_b1-def_b2-def_eps-def_fi-TWICE_CURRENT_SIGNAL_ANY_SIGN.csv"

    # Show pulls when starting at signal values
    run ./bin/plot_pulls.py \
        --write-svg "results/pulls-${model}-CURRENT_SIGNAL-%name%.svg" \
        "results/fit-${model}_rate-0.10_b1-def_b2-def_eps-def_fi-CURRENT_SIGNAL.csv"
done

for csv in $(find results/ -name 'fit-*.csv')
do
    info=${csv:12:-4}
    model=$(echo "${info}" | cut -d '_' -f 1)
    model_len=$(echo -n "${model}" | wc -c)
    info_wo_model=${info:$((${model_len} + 1))}

    run ./bin/plot_confidence.py \
        --write-svg "results/confidence-${model}-%name%-${info_wo_model}.svg" \
        ${csv}
done

for signal_count in 600 2400
do
    run ./bin/plot_q_test_statistic.py \
        --write-svg "results/q_test_statistic-${signal_count}.svg" \
        results/q_test_stat-${signal_count}-SM.txt \
        results/q_test_stat-${signal_count}-NP.txt
done