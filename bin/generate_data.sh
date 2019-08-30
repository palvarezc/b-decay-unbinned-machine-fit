#!/bin/bash
# This is a quick and dirty bash script that generates all data for publication
# Output files are built up in stripes so that some data is available to use quickly

set -e

iteration_step=250
results_dir="results"

fit_iterations_max=1000
fit_models="SM NP"
fit_rates="0.05 0.10 0.15 0.20"
fit_beta1s="0.85 0.95"
fit_beta2s="0.995 0.9995"
fit_epsilons="1e-03 1e-05"

q_stat_iterations_max=5000
q_stat_models="SM NP"
q_stat_signal_counts="600 2400"

is_run_needed() {
    filename=$1
    needed_count=$2
    header_count=$3

    if [[ -f "${filename}" ]]; then
        count=$(($(wc -l ${filename} | awk '{print $1}') - ${header_count}))
        if [[ "${count}" -ge "${needed_count}" ]]; then
            echo "* ${filename}: Already has ${count}/${needed_count} iterations. Skipping"
            echo
            return 1
        fi
        echo "* ${filename}: Already has ${count}/${needed_count} iterations. Resuming"
    else
        echo "* ${filename}: Creating ${filename} for ${needed_count} iterations"
    fi

    return 0
}

fit() {
    iteration=$1
    model=$2
    rate=$3
    beta1=$4
    beta2=$5
    epsilon=$6
    fit_init=$7

    csv="${results_dir}/fit-${model}_rate-${rate}_b1-${beta1}_b2-${beta2}_eps-${epsilon}_fi-${fit_init}.csv"

    is_run_needed "${csv}" "${iteration}" 2 || return 0

    opts=""
    if [[ "${rate}" != "def" ]]; then
        opts="${opts} --learning-rate ${rate}"
    fi
    if [[ "${beta1}" != "def" ]]; then
        opts="${opts} --opt-param beta_1 ${beta1}"
    fi
    if [[ "${beta2}" != "def" ]]; then
        opts="${opts} --opt-param beta_2 ${beta2}"
    fi
    if [[ "${epsilon}" != "def" ]]; then
        opts="${opts} --opt-param epsilon ${epsilon}"
    fi
    if [[ "${fit_init}" != "def" ]]; then
        opts="${opts} --fit-init ${fit_init}"
    fi

    ./bin/fit.py ${opts} --csv ${csv} --iteration ${iteration} --signal-model ${model}
    echo
}

# Go to project directory
cd ${BASH_SOURCE[0]%/*}/..

[[ -d ${results_dir} ]] || mkdir ${results_dir}

# Generate fit CSVs
for iteration in $(seq ${iteration_step} ${iteration_step} ${fit_iterations_max})
do
    for model in ${fit_models}
    do
        for rate in ${fit_rates}
        do
                fit ${iteration} ${model} ${rate} def def def def
        done
        for beta1 in ${fit_beta1s}
        do
                fit ${iteration} ${model} def ${beta1} def def def
        done
        for beta2 in ${fit_beta2s}
        do
                fit ${iteration} ${model} def def ${beta2} def def
        done
        for epsilon in ${fit_epsilons}
        do
                fit ${iteration} ${model} def def def ${epsilon} def
        done
        fit ${iteration} ${model} def def def def TWICE_CURRENT_SIGNAL_ANY_SIGN
        fit ${iteration} ${model} 0.10 def def def CURRENT_SIGNAL
    done
done

# Generate Q test txt files
for iteration in $(seq ${iteration_step} ${iteration_step} ${q_stat_iterations_max})
do
    for dataset_model in ${q_stat_models}
    do
        for signal_count in ${q_stat_signal_counts}
        do
            txt="${results_dir}/q_test_stat-${signal_count}-${dataset_model}.txt"
            is_run_needed "${txt}" "${iteration}" 0 || continue
            ./bin/q_test_statistic.py \
                --txt ${txt} \
                --iteration ${iteration} \
                --signal-count ${signal_count} \
                --signal-model ${dataset_model} \
                --test-model NP \
                --null-model SM
        done
    done
done