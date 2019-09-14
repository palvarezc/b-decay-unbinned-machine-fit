#!/bin/bash
# This is a quick and dirty bash script that generates all data for publication
# Output files are built up in stripes so that some data is available to use quickly

set -o errexit

iteration_step=250
results_dir="results"

rate_default="$(egrep '^[ ]*learning_rate_default' b_meson_fit/optimizer.py | awk -F' = ' '{print $2}')"
beta1_default="0.90"
beta2_default="0.999"
eps_default="1e-08"

fit_iterations_max=1000
fit_models="SM NP"
fit_rates="0.05 0.10 0.15 0.20"
# Defaults are excluded from below
fit_beta1s="0.85 0.95"
fit_beta2s="0.995 0.9995"
fit_epsilons="1e-03 1e-05"

q_stat_iterations_max=5000
q_stat_models="SM NP"
q_stat_signal_counts="600 2400"

run() {
    echo "**************"
    echo "$@"
    "$@"
    echo "**************"
    echo
}

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

    # CSV filename with default values as 'def'
    csv_with_def=$(echo "${results_dir}/fit-${model}_rate-${rate}_b1-${beta1}_b2-${beta2}_eps-${epsilon}_fi-${fit_init}.csv" | sed "s/rate-${rate_default}/rate-def/")
    # CSV filename with default values as their actual float value
    csv_without_def=$(echo ${csv_with_def} | sed "s/rate-def/rate-${rate_default}/" | sed "s/b1-def/b1-${beta1_default}/" | sed "s/b2-def/b2-${beta2_default}/" | sed "s/eps-def/eps-${eps_default}/")

    is_run_needed "${csv_without_def}" "${iteration}" 2 || return 0

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

    run ./bin/fit.py ${opts} --csv ${csv_without_def} --iteration ${iteration} --signal-model ${model}

    # If we're using any defaults then create a symlink to a filename with values replaced with 'def' to make
    #  post-processing easier
    if [[ "${csv_with_def}" != "${csv_without_def}" ]]; then
        ln -sf $(echo ${csv_without_def} | cut -d '/' -f 2) ${csv_with_def}
    fi

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
        fit ${iteration} ${model} 0.005 def def def CURRENT_SIGNAL
    done
done

# Generate Q test txt files
for iteration in $(seq ${iteration_step} ${iteration_step} ${q_stat_iterations_max})
do
    for dataset_model in ${q_stat_models}
    do
        for signal_count in ${q_stat_signal_counts}
        do
            csv="${results_dir}/q_test_stat-${signal_count}-${dataset_model}.csv"
            is_run_needed "${csv}" "${iteration}" 1 || continue
            run ./bin/q_test_statistic.py \
                --csv ${csv} \
                --iteration ${iteration} \
                --signal-count ${signal_count} \
                --signal-model ${dataset_model} \
                --test-model NP \
                --null-model SM \
                --learning-rate 0.10 \
                --opt-param beta_1 0.85 \
                --opt-param epsilon 1e-3
        done
    done
done
