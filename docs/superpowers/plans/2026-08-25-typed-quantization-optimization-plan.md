# Typed Quantization CUDA Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove avoidable host and temporary-buffer overhead from NVFP4 and typed MXFP8/NVFP4 CUDA paths without changing the public API or reference semantics.

**Architecture:** Keep the current synchronous wrappers and FP32 kernels. Add device-side NVFP4 global-scale selection, then add typed kernels that convert at the point of load/store so typed entry points no longer allocate FP32 temporaries. Existing CPU reference and CUDA tests remain the correctness oracle.

**Tech Stack:** CUDA C++17, CMake/Ninja, CTest, CUDA Events, Compute Sanitizer.

---

### Task 1: Establish the optimization baseline

**Files:**
- Read: `02_quant_dequant/HGinkgo/cuda_quant.cu`
- Read: `02_quant_dequant/HGinkgo/test_cuda.cu`
- Read: `02_quant_dequant/HGinkgo/benchmark.cu`

- [x] Run `cmake --build /tmp/low_precision_cuda_build -j2` and `ctest --test-dir /tmp/low_precision_cuda_build --output-on-failure`.
- [x] Run `/tmp/low_precision_cuda_build/benchmark 1024 2048 5` and save the typed MXFP8/NVFP4 p50 rows as the before measurement.

### Task 2: Move NVFP4 global-scale selection to the device

**Files:**
- Modify: `02_quant_dequant/HGinkgo/cuda_quant.cu:267-311,469-508`
- Test: `02_quant_dequant/HGinkgo/test_cuda.cu`

- [x] Add a kernel that reads the reduced device global `amax`, computes the selected NVFP4 global scale with the same `kMxFp8Max * kNvFp4Max` formula, and stores it in the existing one-element device buffer.
- [x] Replace the host `cudaMemcpy`/host computation/device `cudaMemcpy` sequence with the new kernel; copy the final device scale to `global_scale` only after quantization kernels complete when the optional host pointer is non-null.
- [x] Preserve zero-input behavior (`global_scale == 1.0f`) and the exact CPU-reference float value.
- [x] Build and run the CUDA reference test, including the existing global-scale and payload assertions.

### Task 3: Add direct typed quantization and dequantization kernels

**Files:**
- Modify: `02_quant_dequant/HGinkgo/cuda_quant.cu:313-467,510-561`
- Test: `02_quant_dequant/HGinkgo/test_cuda.cu`

- [x] Add FP16-input MXFP8/NVFP4 quantization kernels that call `fp16_to_float_device` per element and reuse the existing scale and packing logic.
- [x] Add FP16-output and BF16-output MXFP8/NVFP4 dequantization kernels that decode to FP32 and convert directly to the requested storage type.
- [x] Replace typed wrapper temporary `DeviceBuffer<float>` allocations with the direct kernels; leave FP32 wrappers unchanged.
- [x] Run the typed CUDA test and Compute Sanitizer, verifying payloads, scales, global scale, FP16 output, and BF16 output against the CPU reference.

### Task 4: Measure and keep only demonstrated gains

**Files:**
- Modify: `.agent/benchmarks.md`
- Read: `02_quant_dequant/HGinkgo/README.md`

- [x] Run `benchmark --sweep 1` and compare typed and NVFP4 API p50 values with the baseline.
- [x] Run `ctest --test-dir /tmp/low_precision_cuda_build --output-on-failure` and `git diff --check`.
- [x] Record the before/after timings and payload mismatch count; do not claim improvement if the device timing is unchanged or worse.

### Task 5: Commit the implementation

- [x] Inspect `git diff --stat` and `git status --short` for only the intended optimization and benchmark record.
- [x] Commit with `perf: fuse typed CUDA conversion paths`.
- [x] Push the current branch after verification.
