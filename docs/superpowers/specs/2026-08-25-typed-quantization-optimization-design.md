# Typed Quantization CUDA Optimization Design

## Goal

Reduce CUDA API overhead for NVFP4 and typed FP16/BF16 paths while preserving the existing public API, CPU-reference payload semantics, and synchronous behavior.

## Scope

- Compute NVFP4 global scale on the device so quantization no longer copies one `amax` scalar to the host and the selected scale back to the device.
- Add direct FP16-input quantization kernels for MXFP8 and NVFP4, avoiding a temporary FP32 input buffer.
- Add direct FP16/BF16-output dequantization kernels for MXFP8 and NVFP4, avoiding a temporary FP32 output buffer.
- Keep the existing FP32 entry points and function signatures unchanged.
- Do not add a workspace or stream API in this change.

## Correctness Contract

The optimized paths must produce byte-identical quantized values, block scales, and NVFP4 global scale compared with the existing CPU reference for nearest rounding. Typed dequantization must match the existing FP32 intermediate conversion semantics for FP16 and BF16 outputs. Existing tail, zero, saturation, and distribution coverage remains mandatory.

## Performance Contract

Benchmark the same `256x256`, `1024x2048`, and `4096x4096` sweep before and after the change. Record API and CUDA-event device p50 timings separately for FP32 and typed paths. Keep the optimization only if correctness is unchanged and the typed path or NVFP4 API path shows a measurable improvement.

## Failure Handling

Kernel launch errors continue to use the existing `check_kernel` path. Device-side global scale storage must be initialized and copied to the host only after the complete quantization call, preserving the current optional `global_scale` output argument.
