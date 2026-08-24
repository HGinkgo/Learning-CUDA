#!/usr/bin/env bash
set -euo pipefail

cli="$1"
workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

printf '\x00\x00\x80\x3f\x00\x00\x00\xc0\x00\x00\x00\x3f\x00\x00\x80\x40' \
  > "${workdir}/input.bin"
"${cli}" quantize --input "${workdir}/input.bin" --output "${workdir}/tensor.qnt" \
  --rows 1 --cols 4 --input-type fp32 --format mxfp8
"${cli}" dequantize --input "${workdir}/tensor.qnt" --output "${workdir}/output.bin" \
  --output-type fp32
test "$(stat -c '%s' "${workdir}/output.bin")" -eq 16
test "$(od -An -tx4 -v "${workdir}/output.bin" | tr -d ' \n')" = \
  "3f800000c00000003f00000040800000"

printf '\x00\x00\x80\x3f\x00\x00\x00\xc0\x00\x00\x00\x3f\x00\x00\x80\x40\x00\x00\x00\x00' \
  > "${workdir}/input_nv.bin"
"${cli}" quantize --input "${workdir}/input_nv.bin" --output "${workdir}/tensor_nv.qnt" \
  --rows 1 --cols 5 --input-type fp32 --format nvfp4
"${cli}" dequantize --input "${workdir}/tensor_nv.qnt" --output "${workdir}/output_bf16.bin" \
  --output-type bf16
test "$(stat -c '%s' "${workdir}/output_bf16.bin")" -eq 10

if "${cli}" quantize --input "${workdir}/input.bin" --output "${workdir}/bad.qnt" \
  --rows 1 --cols 4 --input-type fp32 >/dev/null 2>&1; then
  exit 1
fi
