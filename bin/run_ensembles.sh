#!/bin/bash

set -e

iterations_max=1000
iteration_step=250

models="SM NP"
rates="0.05 0.10 0.15 0.20"
beta1s="0.85 0.95"
beta2s="0.995 0.9995"
epsilons="1e-03 1e-05"

results_dir="results"

fit() {
    iteration=$1
    model=$2
    rate=$3
    beta1=$4
    beta2=$5
    epsilon=$6
    fit_init=$7

    csv="${results_dir}/${model}_rate-${rate}_b1-${beta1}_b2-${beta2}_eps-${epsilon}_fi-${fit_init}.csv"
    if [[ -f "${csv}" ]]; then
        csv_rows=$(($(wc -l ${csv} | awk '{print $1}') - 2))
        if [[ "${csv_rows}" -ge "${iteration}" ]]; then
            echo "* ${csv}: Already has ${csv_rows}/${iteration} iterations. Skipping"
            echo
            return
        fi
        echo "* ${csv}: Already has ${csv_rows}/${iteration} iterations. Resuming"
    else
        echo "* ${csv}: Creating ${csv} for ${iteration} iterations"
    fi

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

for iteration in $(seq ${iteration_step} ${iteration_step} ${iterations_max})
do
    for model in ${models}
    do
        for rate in ${rates}
        do
                fit ${iteration} ${model} ${rate} def def def def
        done
        for beta1 in ${beta1s}
        do
                fit ${iteration} ${model} def ${beta1} def def def
        done
        for beta2 in ${beta2s}
        do
                fit ${iteration} ${model} def def ${beta2} def def
        done
        for epsilon in ${epsilons}
        do
                fit ${iteration} ${model} def def def ${epsilon} def
        done
        fit ${iteration} ${model} def def def def TWICE_CURRENT_SIGNAL_ANY_SIGN
        fit ${iteration} ${model} 0.10 def def def CURRENT_SIGNAL
    done
done
