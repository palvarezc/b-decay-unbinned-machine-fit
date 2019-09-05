#!/bin/bash

if [ "$#" -ne "1" ]; then
	echo "Usage: $0 <csv>"
	exit 1
fi

csv=$1

row_count=$(wc -l <(sed -r '/^[a-z0]+/d' ${csv}) | cut -d ' ' -f 1)
total_time=$(awk -F, '{print $NF}' ${csv} | tr -d '\r' | paste -sd+ | bc -l)

echo "${total_time}/${row_count}" | bc -l
