#include "cuda_quant.hpp"

#include "quant.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace low_precision {
namespace {

constexpr std::size_t kMxBlock = 32;
constexpr std::size_t kNvBlock = 16;
constexpr float kMxFp8Max = 448.0f;
constexpr float kNvFp4Max = 6.0f;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_kernel(const char* operation) {
    check_cuda(cudaGetLastError(), operation);
    check_cuda(cudaDeviceSynchronize(), operation);
}

void check_shape(std::size_t rows, std::size_t cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("matrix dimensions must be nonzero");
    }
}

std::size_t block_count(std::size_t cols, std::size_t block_size) {
    return (cols + block_size - 1) / block_size;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check_cuda(cudaMalloc(&pointer_, count_ * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return pointer_; }

private:
    std::size_t count_;
    T* pointer_ = nullptr;
};

__device__ float decode_e2m1_device(unsigned code) {
    const unsigned magnitude_code = code & 0x07u;
    const unsigned exponent = magnitude_code >> 1u;
    const unsigned mantissa = magnitude_code & 0x01u;
    float value;
    if (exponent == 0) {
        value = static_cast<float>(mantissa) * 0.5f;
    } else {
        value = ldexpf(1.0f + static_cast<float>(mantissa) * 0.5f,
                       static_cast<int>(exponent) - 1);
    }
    return (code & 0x08u) != 0 ? -value : value;
}

__device__ unsigned encode_e2m1_device(float value) {
    const bool negative = signbit(value);
    const float magnitude = fminf(fabsf(value), kNvFp4Max);
    unsigned lower = 0;
    unsigned upper = 7;
    for (unsigned code = 0; code <= 7; ++code) {
        const float candidate = decode_e2m1_device(code);
        if (candidate <= magnitude) {
            lower = code;
        }
        if (candidate >= magnitude) {
            upper = code;
            break;
        }
    }
    unsigned selected = lower;
    const float lower_distance = magnitude - decode_e2m1_device(lower);
    const float upper_distance = decode_e2m1_device(upper) - magnitude;
    if (upper_distance < lower_distance ||
        (upper_distance == lower_distance && (lower & 1u) != 0u)) {
        selected = upper;
    }
    return negative ? selected | 0x08u : selected;
}

__device__ float decode_e4m3_magnitude_device(unsigned code) {
    const unsigned exponent = (code >> 3u) & 0x0fu;
    const unsigned mantissa = code & 0x07u;
    if (exponent == 0) {
        return ldexpf(static_cast<float>(mantissa), -9);
    }
    return ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                  static_cast<int>(exponent) - 7);
}

__device__ float decode_e4m3_device(unsigned code) {
    const float magnitude = decode_e4m3_magnitude_device(code & 0x7fu);
    return (code & 0x80u) != 0 ? -magnitude : magnitude;
}

__device__ unsigned encode_e4m3_device(float value) {
    const bool negative = signbit(value);
    const float magnitude = fminf(fabsf(value), kMxFp8Max);
    unsigned lower = 0;
    unsigned upper = 0x7e;
    for (unsigned code = 0; code <= 0x7e; ++code) {
        const float candidate = decode_e4m3_magnitude_device(code);
        if (candidate <= magnitude) {
            lower = code;
        }
        if (candidate >= magnitude) {
            upper = code;
            break;
        }
    }
    unsigned selected = lower;
    const float lower_distance = magnitude - decode_e4m3_magnitude_device(lower);
    const float upper_distance = decode_e4m3_magnitude_device(upper) - magnitude;
    if (upper_distance < lower_distance ||
        (upper_distance == lower_distance && (lower & 1u) != 0u)) {
        selected = upper;
    }
    return negative ? selected | 0x80u : selected;
}

__device__ float decode_e8m0_device(unsigned code) {
    return ldexpf(1.0f, static_cast<int>(code) - 127);
}

__device__ float fp16_to_float_device(Fp16 value) {
    return __half2float(__ushort_as_half(value.bits));
}

__device__ Fp16 fp16_from_float_device(float value) {
    return {__half_as_ushort(__float2half_rn(value))};
}

__device__ Bf16 bf16_from_float_device(float value) {
    unsigned bits = __float_as_uint(value);
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return {static_cast<std::uint16_t>(bits >> 16u)};
}

__global__ void fp16_to_float_kernel(const Fp16* input, float* output,
                                     std::size_t element_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = fp16_to_float_device(input[index]);
    }
}

__global__ void float_to_fp16_kernel(const float* input, Fp16* output,
                                     std::size_t element_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = fp16_from_float_device(input[index]);
    }
}

__global__ void float_to_bf16_kernel(const float* input, Bf16* output,
                                     std::size_t element_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = bf16_from_float_device(input[index]);
    }
}

unsigned conversion_grid(std::size_t element_count) {
    return static_cast<unsigned>((element_count + 255) / 256);
}

void convert_fp16_to_float(const Fp16* input, float* output, std::size_t element_count) {
    fp16_to_float_kernel<<<conversion_grid(element_count), 256>>>(input, output, element_count);
    check_kernel("FP16 input conversion");
}

void convert_float_to_fp16(const float* input, Fp16* output, std::size_t element_count) {
    float_to_fp16_kernel<<<conversion_grid(element_count), 256>>>(input, output, element_count);
    check_kernel("FP16 output conversion");
}

void convert_float_to_bf16(const float* input, Bf16* output, std::size_t element_count) {
    float_to_bf16_kernel<<<conversion_grid(element_count), 256>>>(input, output, element_count);
    check_kernel("BF16 output conversion");
}

__device__ void reduce_max_32(float* values) {
    for (unsigned stride = 16; stride > 0; stride >>= 1u) {
        __syncthreads();
        if (threadIdx.x < stride) {
            values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
        }
    }
    __syncthreads();
}

__global__ void block_amax_kernel(const float* input,
                                  float* block_amax,
                                  std::size_t rows,
                                  std::size_t cols,
                                  std::size_t block_size,
                                  std::size_t blocks_per_row) {
    const std::size_t block_index = blockIdx.x;
    if (block_index >= rows * blocks_per_row) {
        return;
    }
    const std::size_t row = block_index / blocks_per_row;
    const std::size_t block = block_index % blocks_per_row;
    const std::size_t begin = block * block_size;
    const std::size_t col = begin + threadIdx.x;
    float local = 0.0f;
    if (col < cols && threadIdx.x < block_size) {
        local = fabsf(input[row * cols + col]);
    }
    __shared__ float shared[32];
    shared[threadIdx.x] = local;
    reduce_max_32(shared);
    if (threadIdx.x == 0) {
        block_amax[block_index] = shared[0];
    }
}

__global__ void nvfp4_block_amax_kernel(const float* input,
                                        float* block_amax,
                                        float* global_amax,
                                        std::size_t rows,
                                        std::size_t cols,
                                        std::size_t blocks_per_row) {
    constexpr unsigned kWarpsPerBlock = 8;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5u;
    const std::size_t block_index = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const std::size_t block_count_total = rows * blocks_per_row;
    float local = 0.0f;
    if (block_index < block_count_total && lane < kNvBlock) {
        const std::size_t row = block_index / blocks_per_row;
        const std::size_t block = block_index % blocks_per_row;
        const std::size_t col = block * kNvBlock + lane;
        if (col < cols) {
            local = fabsf(input[row * cols + col]);
        }
    }
    for (unsigned offset = 16; offset > 0; offset >>= 1u) {
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    if (lane == 0 && block_index < block_count_total) {
        block_amax[block_index] = local;
        atomicMax(reinterpret_cast<unsigned int*>(global_amax), __float_as_uint(local));
    }
}

__global__ void encode_nvfp4_scales_kernel(const float* block_amax,
                                           const float* global_scale,
                                           std::uint8_t* block_scales,
                                           std::size_t block_count_total) {
    const std::size_t block_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (block_index >= block_count_total) {
        return;
    }
    const float selected_global_scale = *global_scale;
    const float block_max = block_amax[block_index];
    const float requested_scale = block_max == 0.0f
                                      ? 1.0f
                                      : fmaxf((block_max / kNvFp4Max) / selected_global_scale,
                                              decode_e4m3_magnitude_device(0x01));
    block_scales[block_index] = static_cast<std::uint8_t>(encode_e4m3_device(requested_scale));
}

__global__ void quantize_mxfp8_kernel(const float* input,
                                      std::uint8_t* values,
                                      const std::uint8_t* scales,
                                      std::size_t rows,
                                      std::size_t cols,
                                      std::size_t blocks_per_row) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) {
        return;
    }
    const std::size_t row = index / cols;
    const std::size_t col = index % cols;
    const std::size_t block = col / kMxBlock;
    const float scale = decode_e8m0_device(scales[row * blocks_per_row + block]);
    values[index] = static_cast<std::uint8_t>(encode_e4m3_device(input[index] / scale));
}

__global__ void dequantize_mxfp8_kernel(const std::uint8_t* values,
                                        const std::uint8_t* scales,
                                        float* output,
                                        std::size_t rows,
                                        std::size_t cols,
                                        std::size_t blocks_per_row) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) {
        return;
    }
    const std::size_t row = index / cols;
    const std::size_t col = index % cols;
    const std::size_t block = col / kMxBlock;
    output[index] = decode_e4m3_device(values[index]) *
                    decode_e8m0_device(scales[row * blocks_per_row + block]);
}

__global__ void quantize_nvfp4_kernel(const float* input,
                                      std::uint8_t* values,
                                      const std::uint8_t* block_scales,
                                      const float* global_scale,
                                      std::size_t rows,
                                      std::size_t cols,
                                      std::size_t blocks_per_row) {
    const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t element_count = rows * cols;
    if (pair >= (element_count + 1) / 2) {
        return;
    }
    const std::size_t first = pair * 2;
    const std::size_t second = first + 1;
    const std::size_t first_row = first / cols;
    const std::size_t first_col = first % cols;
    const std::size_t first_block = first_col / kNvBlock;
    const float selected_global_scale = *global_scale;
    const float first_scale = selected_global_scale *
                              decode_e4m3_device(block_scales[first_row * blocks_per_row + first_block]);
    unsigned packed = encode_e2m1_device(input[first] / first_scale);
    if (second < element_count) {
        const std::size_t second_row = second / cols;
        const std::size_t second_col = second % cols;
        const std::size_t second_block = second_col / kNvBlock;
        const float second_scale = selected_global_scale *
                                   decode_e4m3_device(block_scales[second_row * blocks_per_row + second_block]);
        packed |= encode_e2m1_device(input[second] / second_scale) << 4u;
    }
    values[pair] = static_cast<std::uint8_t>(packed);
}

__global__ void dequantize_nvfp4_kernel(const std::uint8_t* values,
                                        const std::uint8_t* block_scales,
                                        float global_scale,
                                        float* output,
                                        std::size_t rows,
                                        std::size_t cols,
                                        std::size_t blocks_per_row) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * cols) {
        return;
    }
    const std::size_t row = index / cols;
    const std::size_t col = index % cols;
    const std::size_t block = col / kNvBlock;
    const unsigned code = (index & 1u) == 0 ? values[index / 2] & 0x0fu : values[index / 2] >> 4u;
    output[index] = decode_e2m1_device(code) * global_scale *
                    decode_e4m3_device(block_scales[row * blocks_per_row + block]);
}

}  // namespace

void cuda_quantize_mxfp8(const float* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_scales,
                         std::size_t rows,
                         std::size_t cols) {
    check_shape(rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kMxBlock);
    DeviceBuffer<float> d_amax(rows * blocks_per_row);
    std::vector<float> h_amax(rows * blocks_per_row);
    const dim3 reduction_grid(static_cast<unsigned>(rows * blocks_per_row));
    block_amax_kernel<<<reduction_grid, 32>>>(d_input, d_amax.get(), rows, cols,
                                              kMxBlock, blocks_per_row);
    check_kernel("MXFP8 block amax");
    check_cuda(cudaMemcpy(h_amax.data(), d_amax.get(), h_amax.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "MXFP8 amax copy");

    std::vector<std::uint8_t> h_scales(h_amax.size());
    for (std::size_t i = 0; i < h_amax.size(); ++i) {
        const float requested_scale = h_amax[i] == 0.0f ? 1.0f : h_amax[i] / kMxFp8Max;
        h_scales[i] = encode_e8m0(requested_scale);
    }
    check_cuda(cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice),
               "MXFP8 scale copy");

    const std::size_t element_count = rows * cols;
    quantize_mxfp8_kernel<<<static_cast<unsigned>((element_count + 255) / 256), 256>>>(
        d_input, d_values, d_scales, rows, cols, blocks_per_row);
    check_kernel("MXFP8 quantize");
}

void cuda_quantize_mxfp8(const Fp16* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_scales,
                         std::size_t rows,
                         std::size_t cols) {
    check_shape(rows, cols);
    DeviceBuffer<float> converted(rows * cols);
    convert_fp16_to_float(d_input, converted.get(), rows * cols);
    cuda_quantize_mxfp8(converted.get(), d_values, d_scales, rows, cols);
}

void cuda_dequantize_mxfp8(const std::uint8_t* d_values,
                           const std::uint8_t* d_scales,
                           float* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    check_shape(rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kMxBlock);
    const std::size_t element_count = rows * cols;
    dequantize_mxfp8_kernel<<<static_cast<unsigned>((element_count + 255) / 256), 256>>>(
        d_values, d_scales, d_output, rows, cols, blocks_per_row);
    check_kernel("MXFP8 dequantize");
}

void cuda_dequantize_mxfp8(const std::uint8_t* d_values,
                           const std::uint8_t* d_scales,
                           Fp16* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    check_shape(rows, cols);
    DeviceBuffer<float> converted(rows * cols);
    cuda_dequantize_mxfp8(d_values, d_scales, converted.get(), rows, cols);
    convert_float_to_fp16(converted.get(), d_output, rows * cols);
}

void cuda_dequantize_mxfp8(const std::uint8_t* d_values,
                           const std::uint8_t* d_scales,
                           Bf16* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    check_shape(rows, cols);
    DeviceBuffer<float> converted(rows * cols);
    cuda_dequantize_mxfp8(d_values, d_scales, converted.get(), rows, cols);
    convert_float_to_bf16(converted.get(), d_output, rows * cols);
}

void cuda_quantize_nvfp4(const float* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_block_scales,
                         std::size_t rows,
                         std::size_t cols,
                         float* global_scale) {
    check_shape(rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kNvBlock);
    const std::size_t block_count_total = rows * blocks_per_row;
    DeviceBuffer<float> d_amax(block_count_total);
    DeviceBuffer<float> d_global_amax(1);
    DeviceBuffer<float> d_global_scale(1);
    check_cuda(cudaMemset(d_global_amax.get(), 0, sizeof(float)), "NVFP4 global amax reset");
    const unsigned reduction_blocks = static_cast<unsigned>((block_count_total + 7) / 8);
    nvfp4_block_amax_kernel<<<reduction_blocks, 256>>>(
        d_input, d_amax.get(), d_global_amax.get(), rows, cols, blocks_per_row);
    check_kernel("NVFP4 block amax");
    float global_amax = 0.0f;
    check_cuda(cudaMemcpy(&global_amax, d_global_amax.get(), sizeof(float),
                          cudaMemcpyDeviceToHost),
               "NVFP4 global amax copy");
    const float selected_global_scale = global_amax == 0.0f
                                            ? 1.0f
                                            : global_amax / (kMxFp8Max * kNvFp4Max);
    check_cuda(cudaMemcpy(d_global_scale.get(), &selected_global_scale, sizeof(float),
                          cudaMemcpyHostToDevice),
               "NVFP4 global scale copy");
    if (global_scale != nullptr) {
        *global_scale = selected_global_scale;
    }

    encode_nvfp4_scales_kernel<<<static_cast<unsigned>((block_count_total + 255) / 256), 256>>>(
        d_amax.get(), d_global_scale.get(), d_block_scales, block_count_total);
    check_kernel("NVFP4 block scale encode");

    const std::size_t packed_count = (rows * cols + 1) / 2;
    quantize_nvfp4_kernel<<<static_cast<unsigned>((packed_count + 255) / 256), 256>>>(
        d_input, d_values, d_block_scales, d_global_scale.get(), rows, cols, blocks_per_row);
    check_kernel("NVFP4 quantize");
}

void cuda_quantize_nvfp4(const Fp16* d_input,
                         std::uint8_t* d_values,
                         std::uint8_t* d_block_scales,
                         std::size_t rows,
                         std::size_t cols,
                         float* global_scale) {
    check_shape(rows, cols);
    DeviceBuffer<float> converted(rows * cols);
    convert_fp16_to_float(d_input, converted.get(), rows * cols);
    cuda_quantize_nvfp4(converted.get(), d_values, d_block_scales,
                        rows, cols, global_scale);
}

void cuda_dequantize_nvfp4(const std::uint8_t* d_values,
                           const std::uint8_t* d_block_scales,
                           float global_scale,
                           float* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    check_shape(rows, cols);
    const std::size_t blocks_per_row = block_count(cols, kNvBlock);
    const std::size_t element_count = rows * cols;
    dequantize_nvfp4_kernel<<<static_cast<unsigned>((element_count + 255) / 256), 256>>>(
        d_values, d_block_scales, global_scale, d_output, rows, cols, blocks_per_row);
    check_kernel("NVFP4 dequantize");
}

void cuda_dequantize_nvfp4(const std::uint8_t* d_values,
                           const std::uint8_t* d_block_scales,
                           float global_scale,
                           Fp16* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    check_shape(rows, cols);
    DeviceBuffer<float> converted(rows * cols);
    cuda_dequantize_nvfp4(d_values, d_block_scales, global_scale,
                          converted.get(), rows, cols);
    convert_float_to_fp16(converted.get(), d_output, rows * cols);
}

void cuda_dequantize_nvfp4(const std::uint8_t* d_values,
                           const std::uint8_t* d_block_scales,
                           float global_scale,
                           Bf16* d_output,
                           std::size_t rows,
                           std::size_t cols) {
    check_shape(rows, cols);
    DeviceBuffer<float> converted(rows * cols);
    cuda_dequantize_nvfp4(d_values, d_block_scales, global_scale,
                          converted.get(), rows, cols);
    convert_float_to_bf16(converted.get(), d_output, rows * cols);
}

}  // namespace low_precision
