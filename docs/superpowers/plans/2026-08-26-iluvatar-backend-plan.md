# Iluvatar Backend Implementation Plan

> **For agentic workers:** Execute inline with verification checkpoints.

**Goal:** Add an optional Iluvatar CoreX build path for the current low-precision kernels.

**Architecture:** Preserve common headers, CPU reference, file I/O, tests, and public target names. Add a thin Iluvatar source entry that reuses the CUDA implementation, then compile it as `ivcore` with CoreX Clang and link IX-ML's CUDA-compatible runtime.

### Task 1: Add backend selection to CMake

- [x] Add `LP_BACKEND` and `ILUVATAR_SDK_ROOT` cache options.
- [x] Keep NVIDIA's existing CUDA language path as the default.
- [x] Add the Iluvatar source entry, configure CoreX Clang as the CXX/CUDA compiler pair, select `ivcore11`, and link the CUDA-compatible runtime.

### Task 2: Add Iluvatar environment documentation

- [x] Document the CoreX setup command, CMake invocation, runtime library path, and GPU query command in the project README.

### Task 3: Verify on MR-V100

- [x] Configure and build with CoreX Clang.
- [x] Run CTest and the CUDA reference test.
- [x] Check sanitizer/tool availability; no `ix-sanitizer` or `compute-sanitizer` is installed, while `ixgdb` is available.
- [x] Run a benchmark and record payload mismatches and timing.
