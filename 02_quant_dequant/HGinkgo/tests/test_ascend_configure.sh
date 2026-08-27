#!/usr/bin/env bash

set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

source_dir=${1:?source directory is required}
build_dir=${2:-"${TMPDIR:-/tmp}/low-precision-ascend"}

cmake -S "${source_dir}" -B "${build_dir}" -G "Unix Makefiles" \
    -DLP_BACKEND=ascend \
    -DLP_ASCEND_SOC=Ascend910B1
cmake --build "${build_dir}" --target test_cuda cuda_quant_cli benchmark -j2
"${build_dir}/test_cuda"
"${build_dir}/benchmark" 32 33 1 | grep -q '^random,mxfp8,ascend,'
