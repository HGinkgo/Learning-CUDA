# Iluvatar Backend Design

**Goal:** Build the existing MXFP8/NVFP4 CUDA-style kernels with Iluvatar CoreX and run the same correctness and benchmark contract on an Iluvatar GPU.

**Approach:** Keep the public low-precision API and packed payload format unchanged. Add an Iluvatar backend target that compiles the existing kernel source with CoreX Clang's `-x ivcore` language mode and links the IX-ML CUDA-compatible runtime. NVIDIA remains the default backend and its build path is unchanged.

**Build contract:**

- `LP_BACKEND=nvidia`: existing CXX/CUDA CMake path.
- `LP_BACKEND=iluvatar`: CMake uses CoreX Clang for both the host and CUDA language entries; `ILUVATAR_SDK_ROOT` defaults to `/usr/local/corex-${COREX_VERSION}` or `/usr/local/corex-4.4.0`.
- Iluvatar kernel source is compiled as `ivcore` while host sources remain ordinary C++17.
- Runtime links against `${ILUVATAR_SDK_ROOT}/lib64/libcudart.so` and uses the SDK library path at runtime.

**Acceptance:** The Iluvatar build runs the existing CPU tests, CUDA reference tests, CLI round-trip, and benchmark smoke test. The GPU reference test must pass on MR-V100, and benchmark output must report zero payload mismatches for MXFP8 and NVFP4.
