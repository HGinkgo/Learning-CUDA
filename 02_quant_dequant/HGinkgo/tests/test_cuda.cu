#include "low_precision/cuda_quant.hpp"
#include "low_precision/quant.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void check_cuda(cudaError_t status) {
    assert(status == cudaSuccess);
}

template <typename T>
T* allocate_device(std::size_t count) {
    T* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, count * sizeof(T)));
    return pointer;
}

void test_mxfp8() {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 35;
    std::vector<float> input(rows * cols, 0.0f);
    input[0] = 448.0f;
    input[1] = 1.0f;
    input[32] = -3.0f;
    input[35] = -448.0f;
    input[69] = 0.5f;

    const auto expected = low_precision::quantize_mxfp8(input, rows, cols);
    float* d_input = allocate_device<float>(input.size());
    std::uint8_t* d_values = allocate_device<std::uint8_t>(input.size());
    std::uint8_t* d_scales = allocate_device<std::uint8_t>(expected.scales.size());
    check_cuda(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    low_precision::cuda_quantize_mxfp8(d_input, d_values, d_scales, rows, cols);

    std::vector<std::uint8_t> values(input.size());
    std::vector<std::uint8_t> scales(expected.scales.size());
    check_cuda(cudaMemcpy(values.data(), d_values, values.size(), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(scales.data(), d_scales, scales.size(), cudaMemcpyDeviceToHost));
    assert(values == expected.values);
    assert(scales == expected.scales);

    std::vector<float> decoded(input.size());
    float* d_decoded = allocate_device<float>(decoded.size());
    low_precision::cuda_dequantize_mxfp8(d_values, d_scales, d_decoded, rows, cols);
    check_cuda(cudaMemcpy(decoded.data(), d_decoded, decoded.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const auto expected_decoded = low_precision::dequantize_mxfp8(expected);
    for (std::size_t i = 0; i < decoded.size(); ++i) {
        assert(std::fabs(decoded[i] - expected_decoded[i]) <= 1e-6f);
    }
    check_cuda(cudaFree(d_decoded));
    check_cuda(cudaFree(d_scales));
    check_cuda(cudaFree(d_values));
    check_cuda(cudaFree(d_input));
}

void test_nvfp4() {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 17;
    std::vector<float> input(rows * cols, 0.0f);
    input[0] = 6.0f;
    input[1] = -6.0f;
    input[16] = 1.0f;
    input[17] = -1.0f;
    input[33] = 0.5f;

    const auto expected = low_precision::quantize_nvfp4(input, rows, cols);
    float* d_input = allocate_device<float>(input.size());
    std::uint8_t* d_values = allocate_device<std::uint8_t>(expected.values.size());
    std::uint8_t* d_scales = allocate_device<std::uint8_t>(expected.block_scales.size());
    check_cuda(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    float global_scale = 0.0f;
    low_precision::cuda_quantize_nvfp4(d_input, d_values, d_scales, rows, cols,
                                       &global_scale);
    std::vector<std::uint8_t> values(expected.values.size());
    std::vector<std::uint8_t> scales(expected.block_scales.size());
    check_cuda(cudaMemcpy(values.data(), d_values, values.size(), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(scales.data(), d_scales, scales.size(), cudaMemcpyDeviceToHost));
    assert(values == expected.values);
    assert(scales == expected.block_scales);
    assert(std::fabs(global_scale - expected.global_scale) <= 1e-7f);

    std::vector<float> decoded(input.size());
    float* d_decoded = allocate_device<float>(decoded.size());
    low_precision::cuda_dequantize_nvfp4(d_values, d_scales, global_scale,
                                         d_decoded, rows, cols);
    check_cuda(cudaMemcpy(decoded.data(), d_decoded, decoded.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const auto expected_decoded = low_precision::dequantize_nvfp4(expected);
    for (std::size_t i = 0; i < decoded.size(); ++i) {
        assert(std::fabs(decoded[i] - expected_decoded[i]) <= 1e-6f);
    }
    check_cuda(cudaFree(d_decoded));
    check_cuda(cudaFree(d_scales));
    check_cuda(cudaFree(d_values));
    check_cuda(cudaFree(d_input));
}

void test_nvfp4_odd_matrix() {
    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 33;
    std::vector<float> input(rows * cols);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.375f;
    }
    input[32] = 6.0f;
    input[33] = -6.0f;
    input[98] = 0.5f;

    const auto expected = low_precision::quantize_nvfp4(input, rows, cols);
    float* d_input = allocate_device<float>(input.size());
    std::uint8_t* d_values = allocate_device<std::uint8_t>(expected.values.size());
    std::uint8_t* d_scales = allocate_device<std::uint8_t>(expected.block_scales.size());
    check_cuda(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

    float global_scale = 0.0f;
    low_precision::cuda_quantize_nvfp4(d_input, d_values, d_scales, rows, cols,
                                       &global_scale);
    std::vector<std::uint8_t> values(expected.values.size());
    std::vector<std::uint8_t> scales(expected.block_scales.size());
    check_cuda(cudaMemcpy(values.data(), d_values, values.size(), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(scales.data(), d_scales, scales.size(), cudaMemcpyDeviceToHost));
    assert(values == expected.values);
    assert(scales == expected.block_scales);
    assert(std::fabs(global_scale - expected.global_scale) <= 1e-7f);

    std::vector<float> decoded(input.size());
    float* d_decoded = allocate_device<float>(decoded.size());
    low_precision::cuda_dequantize_nvfp4(d_values, d_scales, global_scale,
                                         d_decoded, rows, cols);
    check_cuda(cudaMemcpy(decoded.data(), d_decoded, decoded.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const auto expected_decoded = low_precision::dequantize_nvfp4(expected);
    for (std::size_t index = 0; index < decoded.size(); ++index) {
        assert(std::fabs(decoded[index] - expected_decoded[index]) <= 1e-6f);
    }

    check_cuda(cudaFree(d_decoded));
    check_cuda(cudaFree(d_scales));
    check_cuda(cudaFree(d_values));
    check_cuda(cudaFree(d_input));
}

void test_fp16_input_and_low_precision_outputs() {
    constexpr std::size_t mxfp8_rows = 1;
    constexpr std::size_t mxfp8_cols = 35;
    std::vector<float> float_input(mxfp8_cols);
    for (std::size_t index = 0; index < float_input.size(); ++index) {
        float_input[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.75f;
    }
    std::vector<low_precision::Fp16> half_input;
    half_input.reserve(float_input.size());
    for (const float value : float_input) {
        half_input.push_back(low_precision::fp16_from_float(value));
    }
    const auto expected_mx = low_precision::quantize_mxfp8(half_input,
                                                            mxfp8_rows, mxfp8_cols);
    auto* d_mx_input = allocate_device<low_precision::Fp16>(half_input.size());
    auto* d_mx_values = allocate_device<std::uint8_t>(expected_mx.values.size());
    auto* d_mx_scales = allocate_device<std::uint8_t>(expected_mx.scales.size());
    check_cuda(cudaMemcpy(d_mx_input, half_input.data(), half_input.size() * sizeof(half_input[0]),
                          cudaMemcpyHostToDevice));
    low_precision::cuda_quantize_mxfp8(d_mx_input, d_mx_values, d_mx_scales,
                                       mxfp8_rows, mxfp8_cols);
    std::vector<std::uint8_t> mx_values(expected_mx.values.size());
    std::vector<std::uint8_t> mx_scales(expected_mx.scales.size());
    check_cuda(cudaMemcpy(mx_values.data(), d_mx_values, mx_values.size(), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(mx_scales.data(), d_mx_scales, mx_scales.size(), cudaMemcpyDeviceToHost));
    assert(mx_values == expected_mx.values);
    assert(mx_scales == expected_mx.scales);
    auto* d_mx_half = allocate_device<low_precision::Fp16>(half_input.size());
    auto* d_mx_bfloat = allocate_device<low_precision::Bf16>(half_input.size());
    low_precision::cuda_dequantize_mxfp8(d_mx_values, d_mx_scales, d_mx_half,
                                         mxfp8_rows, mxfp8_cols);
    low_precision::cuda_dequantize_mxfp8(d_mx_values, d_mx_scales, d_mx_bfloat,
                                         mxfp8_rows, mxfp8_cols);
    std::vector<low_precision::Fp16> mx_half(half_input.size());
    std::vector<low_precision::Bf16> mx_bfloat(half_input.size());
    check_cuda(cudaMemcpy(mx_half.data(), d_mx_half, mx_half.size() * sizeof(mx_half[0]),
                          cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(mx_bfloat.data(), d_mx_bfloat,
                          mx_bfloat.size() * sizeof(mx_bfloat[0]), cudaMemcpyDeviceToHost));
    assert(mx_half == low_precision::dequantize_mxfp8_fp16(expected_mx));
    assert(mx_bfloat == low_precision::dequantize_mxfp8_bf16(expected_mx));
    check_cuda(cudaFree(d_mx_bfloat));
    check_cuda(cudaFree(d_mx_half));
    check_cuda(cudaFree(d_mx_scales));
    check_cuda(cudaFree(d_mx_values));
    check_cuda(cudaFree(d_mx_input));

    constexpr std::size_t nvfp4_cols = 17;
    half_input.resize(nvfp4_cols);
    for (std::size_t index = 0; index < half_input.size(); ++index) {
        half_input[index] = low_precision::fp16_from_float(
            static_cast<float>(static_cast<int>(index % 7) - 3) * 0.5f);
    }
    const auto expected_nv = low_precision::quantize_nvfp4(half_input, 1, nvfp4_cols);
    auto* d_nv_input = allocate_device<low_precision::Fp16>(half_input.size());
    auto* d_nv_values = allocate_device<std::uint8_t>(expected_nv.values.size());
    auto* d_nv_scales = allocate_device<std::uint8_t>(expected_nv.block_scales.size());
    check_cuda(cudaMemcpy(d_nv_input, half_input.data(), half_input.size() * sizeof(half_input[0]),
                          cudaMemcpyHostToDevice));
    float global_scale = 0.0f;
    low_precision::cuda_quantize_nvfp4(d_nv_input, d_nv_values, d_nv_scales,
                                       1, nvfp4_cols, &global_scale);
    std::vector<std::uint8_t> nv_values(expected_nv.values.size());
    std::vector<std::uint8_t> nv_scales(expected_nv.block_scales.size());
    check_cuda(cudaMemcpy(nv_values.data(), d_nv_values, nv_values.size(), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(nv_scales.data(), d_nv_scales, nv_scales.size(), cudaMemcpyDeviceToHost));
    assert(nv_values == expected_nv.values);
    assert(nv_scales == expected_nv.block_scales);
    assert(std::fabs(global_scale - expected_nv.global_scale) <= 1e-7f);
    auto* d_nv_half = allocate_device<low_precision::Fp16>(half_input.size());
    auto* d_nv_bfloat = allocate_device<low_precision::Bf16>(half_input.size());
    low_precision::cuda_dequantize_nvfp4(d_nv_values, d_nv_scales, global_scale,
                                         d_nv_half, 1, nvfp4_cols);
    low_precision::cuda_dequantize_nvfp4(d_nv_values, d_nv_scales, global_scale,
                                         d_nv_bfloat, 1, nvfp4_cols);
    std::vector<low_precision::Fp16> nv_half(half_input.size());
    std::vector<low_precision::Bf16> nv_bfloat(half_input.size());
    check_cuda(cudaMemcpy(nv_half.data(), d_nv_half, nv_half.size() * sizeof(nv_half[0]),
                          cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(nv_bfloat.data(), d_nv_bfloat,
                          nv_bfloat.size() * sizeof(nv_bfloat[0]), cudaMemcpyDeviceToHost));
    assert(nv_half == low_precision::dequantize_nvfp4_fp16(expected_nv));
    assert(nv_bfloat == low_precision::dequantize_nvfp4_bf16(expected_nv));
    check_cuda(cudaFree(d_nv_bfloat));
    check_cuda(cudaFree(d_nv_half));
    check_cuda(cudaFree(d_nv_scales));
    check_cuda(cudaFree(d_nv_values));
    check_cuda(cudaFree(d_nv_input));
}

}  // namespace

int main() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cout << "CUDA reference tests skipped: no device\n";
        return 0;
    }
    test_mxfp8();
    test_nvfp4();
    test_nvfp4_odd_matrix();
    test_fp16_input_and_low_precision_outputs();
    std::cout << "CUDA reference tests passed\n";
}
