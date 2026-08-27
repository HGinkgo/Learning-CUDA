#!/usr/bin/env bash
set -euo pipefail

project_root="$1"

set --
set +u
source /usr/local/birensupa/br_container_tools/brsw_set_env.sh
set -u

build_dir="$(mktemp -d)"
trap 'rm -rf "${build_dir}"' EXIT

cmake -S "${project_root}" -B "${build_dir}" -G Ninja \
    -DLP_BACKEND=biren \
    -DLP_BIREN_ARCH=br100

test -n "$(grep '^CMAKE_SUPA_COMPILER:FILEPATH=' "${build_dir}/CMakeCache.txt")"
cmake --build "${build_dir}" --target test_cuda
cmake --build "${build_dir}" --target cuda_quant_cli
cmake --build "${build_dir}" --target benchmark
"${build_dir}/test_cuda"
"${build_dir}/benchmark" 32 33 1 | grep -q '^random,mxfp8,biren,'
