#!/usr/bin/env bash
set -euo pipefail

benchmark="$1"
output="$(${benchmark} 32 33 3 2)"
header="$(printf '%s\n' "${output}" | awk '/^distribution,/{print; exit}')"

printf '%s\n' "${header}" | grep -q 'api_quant_p50_ms'
printf '%s\n' "${header}" | grep -q 'device_quant_p50_ms'
printf '%s\n' "${header}" | grep -q 'api_quant_std_ms'
printf '%s\n' "${header}" | grep -q 'input_type'
printf '%s\n' "${header}" | grep -q 'output_type'

data_rows="$(printf '%s\n' "${output}" | awk -F, '/^(random|normal|outlier),/{print}')"
test "$(printf '%s\n' "${data_rows}" | wc -l)" -eq 24
test "$(printf '%s\n' "${data_rows}" | awk -F, '$NF != 0 {bad++} END {print bad + 0}')" -eq 0
test "$(printf '%s\n' "${data_rows}" | awk -F, '$4 == "fp16" && $5 == "bf16" {found++} END {print found + 0}')" -eq 6
