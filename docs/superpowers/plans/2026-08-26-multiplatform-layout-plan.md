# Multiplatform Backend Layout Implementation Plan

> **For agentic workers:** Execute this plan inline with verification checkpoints; the refactor must preserve the existing NVIDIA payload and benchmark behavior.

**Goal:** Reorganize the low-precision project so common code is separated from the NVIDIA backend and future Iluvatar, MetaX, and Moore Threads backends can be added without duplicating the project.

**Architecture:** Public format and I/O headers live in `include/low_precision`; CPU reference and serialization live in `common`; vendor-specific kernels live in `backends/<vendor>`; executable wrappers and tests live in `apps` and `tests`. The current NVIDIA implementation remains the only enabled backend and keeps its existing API and output format.

**Tech Stack:** CMake, C++17, CUDA C++17, CTest, Bash.

---

### Task 1: Create the stable directory boundaries

**Files:**
- Create: `include/low_precision/`
- Create: `common/`
- Create: `backends/nvidia/`
- Create: `apps/`
- Create: `tests/`

- [x] Move existing headers, implementations, applications, and tests into their new ownership directories without changing file contents.

### Task 2: Update the CMake target graph

**Files:**
- Modify: `CMakeLists.txt`

- [x] Point library, executable, and test sources at the new paths.
- [x] Export `include/` and the NVIDIA backend include path to consumers.
- [x] Keep `CMAKE_CUDA_ARCHITECTURES`, target names, CTest names, and skip behavior unchanged.

### Task 3: Update user-facing paths

**Files:**
- Modify: `README.md`
- Modify: `apps/benchmark_cli.sh` if its self-relative paths require adjustment.
- Modify: `tests/*.sh` only where script locations change.

- [x] Update build, benchmark, CLI, and test commands to use the new paths while keeping commands run from the build directory valid.
- [x] Document the future backend names without adding unimplemented vendor code.

### Task 4: Verify the refactor

- [x] Configure and build a fresh NVIDIA build with CMake/Ninja.
- [x] Run all CTest cases and the CUDA CLI round-trip.
- [x] Run Compute Sanitizer on `test_cuda`.
- [x] Run the 1024x2048 benchmark and confirm zero payload mismatches.
- [x] Run `git diff --check` and confirm only the intended layout/docs files changed.
