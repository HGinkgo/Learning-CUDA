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

MXFP8 的 block scale 编码以及 NVFP4 的 block `amax`、global `amax`、block scale 编码和 packed payload 均在 CUDA 上完成；NVFP4 host 端只往返一个 global scale。

FP16/BF16 CUDA 接口当前在边界处执行格式转换，核心量化和解量化 kernel 继续使用 FP32 中间值，保证与 CPU reference 的 payload 逐字节一致。类型转换是否需要进一步融合，以后续 benchmark 和 profiler 数据为准。
