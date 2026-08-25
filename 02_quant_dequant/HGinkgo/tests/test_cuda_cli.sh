#!/usr/bin/env bash
set -euo pipefail

cli="$1"
cpu_cli="$2"
workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

printf '\x00\x00\x80\x3f\x00\x00\x00\xc0\x00\x00\x00\x3f\x00\x00\x80\x40' \
  > "${workdir}/input.bin"

"${cli}" quantize --input "${workdir}/input.bin" --output "${workdir}/cuda.qnt" \
  --rows 1 --cols 4 --input-type fp32 --format mxfp8 --rounding nearest
"${cpu_cli}" quantize --input "${workdir}/input.bin" --output "${workdir}/cpu.qnt" \
  --rows 1 --cols 4 --input-type fp32 --format mxfp8 --rounding nearest
cmp "${workdir}/cuda.qnt" "${workdir}/cpu.qnt"

"${cli}" dequantize --input "${workdir}/cuda.qnt" --output "${workdir}/output.bin" \
  --output-type fp32
test "$(stat -c '%s' "${workdir}/output.bin")" -eq 16

printf '\x00\x3c\x00\xc0\x00\x38\x00\x44' > "${workdir}/input_fp16.bin"
"${cli}" quantize --input "${workdir}/input_fp16.bin" --output "${workdir}/cuda_fp16.qnt" \
  --rows 1 --cols 4 --input-type fp16 --format nvfp4 --rounding nearest
"${cpu_cli}" quantize --input "${workdir}/input_fp16.bin" --output "${workdir}/cpu_fp16.qnt" \
  --rows 1 --cols 4 --input-type fp16 --format nvfp4 --rounding nearest
cmp "${workdir}/cuda_fp16.qnt" "${workdir}/cpu_fp16.qnt"
"${cli}" dequantize --input "${workdir}/cuda_fp16.qnt" --output "${workdir}/output_fp16.bin" \
  --output-type fp16
"${cli}" dequantize --input "${workdir}/cuda_fp16.qnt" --output "${workdir}/output_bf16.bin" \
  --output-type bf16
test "$(stat -c '%s' "${workdir}/output_fp16.bin")" -eq 8
test "$(stat -c '%s' "${workdir}/output_bf16.bin")" -eq 8

printf '\x00\x00\x80\x3f\x00\x00\x00\xc0\x00\x00\x00\x3f\x00\x00\x80\x40\x00\x00\x00\x00' \
  > "${workdir}/input_nv.bin"
"${cli}" quantize --input "${workdir}/input_nv.bin" --output "${workdir}/cuda_nv.qnt" \
  --rows 1 --cols 5 --input-type fp32 --format nvfp4 --rounding nearest
"${cpu_cli}" quantize --input "${workdir}/input_nv.bin" --output "${workdir}/cpu_nv.qnt" \
  --rows 1 --cols 5 --input-type fp32 --format nvfp4 --rounding nearest
cmp "${workdir}/cuda_nv.qnt" "${workdir}/cpu_nv.qnt"
"${cli}" quantize --input "${workdir}/input_nv.bin" \
  --output "${workdir}/stochastic_nv_a.qnt" --rows 1 --cols 5 --input-type fp32 \
  --format nvfp4 --rounding stochastic --seed 29
"${cli}" quantize --input "${workdir}/input_nv.bin" \
  --output "${workdir}/stochastic_nv_b.qnt" --rows 1 --cols 5 --input-type fp32 \
  --format nvfp4 --rounding stochastic --seed 29
cmp "${workdir}/stochastic_nv_a.qnt" "${workdir}/stochastic_nv_b.qnt"

printf '\xcd\xcc\x8c\x3f\x33\x33\x13\xc0\xa4\x70\xbd\x3e\x66\x66\x96\x40' \
  > "${workdir}/input_stochastic.bin"
"${cli}" quantize --input "${workdir}/input_stochastic.bin" \
  --output "${workdir}/stochastic_a.qnt" --rows 1 --cols 4 --input-type fp32 \
  --format mxfp8 --rounding stochastic --seed 17
"${cli}" quantize --input "${workdir}/input_stochastic.bin" \
  --output "${workdir}/stochastic_b.qnt" --rows 1 --cols 4 --input-type fp32 \
  --format mxfp8 --rounding stochastic --seed 17
cmp "${workdir}/stochastic_a.qnt" "${workdir}/stochastic_b.qnt"
"${cli}" quantize --input "${workdir}/input_stochastic.bin" \
  --output "${workdir}/nearest_stochastic.qnt" --rows 1 --cols 4 --input-type fp32 \
  --format mxfp8 --rounding nearest
if cmp -s "${workdir}/nearest_stochastic.qnt" "${workdir}/stochastic_a.qnt"; then
  exit 1
fi
