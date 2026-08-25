# MXFP8 / NVFP4 Reference

当前目录实现不依赖新硬件的 CPU reference：

- MXFP8：E4M3 数据，按每行连续 32 个元素使用一个 E8M0 scale。
- NVFP4：E2M1 数据，按每行连续 16 个元素使用一个 E4M3 block scale，并保存 FP32 global scale。
- NVFP4 每两个 4-bit 值打包到一个字节，偶数索引使用低 4 bit。
- `NearestEven` 和 `Stochastic` 均提供接口；默认测试使用 `NearestEven`。
- `Fp16` 和 `Bf16` 使用明确的 16-bit storage wrapper；FP16 输入可直接量化，MXFP8/NVFP4 解量化可输出 FP32、FP16 或 BF16。

构建和测试：

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

CLI 使用 little-endian 原始输入和量化文件：

```bash
./build/quant_cli quantize --input input.bin --output tensor.qnt \
  --rows 1024 --cols 2048 --input-type fp32 --format nvfp4 \
  --rounding nearest
./build/quant_cli dequantize --input tensor.qnt --output output.bin \
  --output-type bf16
```

`.qnt` 文件头保存 magic、版本、矩阵尺寸、输入类型、量化格式、rounding、payload 大小和 NVFP4 global scale；读取时会校验 payload 尺寸并拒绝截断或尾随数据。

运行 GPU 正确性和性能测试（默认 `1024 x 2048`，每个 case 重复 5 次）：

```bash
./build/benchmark [rows] [cols] [repeats]
```

运行覆盖小、中、大矩阵的尺寸 sweep：

```bash
./build/benchmark --sweep [repeats]
```

输出为 CSV，包含随机、正态、离群值输入的误差、压缩率、读写字节数、API wall time、CUDA Event device time、p50/min/std、有效带宽和 payload 一致性。CPU 行的 device time 及对应带宽填 0；GPU 行同时保留 API 端到端时间和 device-only 时间。

CSV 还包含 `input_type` 和 `output_type` 列；CUDA benchmark 会额外测量 FP16 输入到 FP16/BF16 输出的路径。CLI 文件读写端到端计时可运行：

```bash
./benchmark_cli.sh ./build/quant_cli 1024 2048
```

真正走 CUDA kernel 的文件工作流使用 `cuda_quant_cli`：

```bash
./build/cuda_quant_cli quantize --input input.bin --output tensor.qnt \
  --rows 1024 --cols 2048 --input-type fp16 --format nvfp4 --rounding nearest
./build/cuda_quant_cli dequantize --input tensor.qnt --output output.bin \
  --output-type bf16
```

该 CLI 将输入复制到 GPU，调用 CUDA 量化/反量化 API，再将 packed payload、scale 或输出张量写回文件。当前 CUDA 量化文件路径只接受 `nearest`；CPU reference CLI 仍提供 `stochastic` 作为对照实现。

MXFP8 的 block scale 编码以及 NVFP4 的 block `amax`、global `amax`、block scale 编码和 packed payload 均在 CUDA 上完成；NVFP4 host 端只往返一个 global scale。

FP16 输入以及 FP16/BF16 输出路径在 CUDA kernel 内直接完成格式转换，不再分配 FP32 临时输入或输出缓冲区；核心计算仍使用 FP32 中间值，保证与 CPU reference 的 payload 逐字节一致。NVFP4 的 global scale 选择也在设备端完成，公开同步 API 只在需要返回 `global_scale` 时复制一个标量到主机。
