#include "quant.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

void expect_close(float actual, float expected, float tolerance = 1e-3f) {
    assert(std::fabs(actual - expected) <= tolerance);
}

void test_storage_round_trip() {
    const low_precision::Fp16 half = low_precision::fp16_from_float(1.5f);
    const low_precision::Bf16 bfloat = low_precision::bf16_from_float(1.5f);
    expect_close(low_precision::fp16_to_float(half), 1.5f);
    expect_close(low_precision::bf16_to_float(bfloat), 1.5f);

    const low_precision::Fp16 largest_half = low_precision::fp16_from_float(65504.0f);
    expect_close(low_precision::fp16_to_float(largest_half), 65504.0f, 1.0f);
    const low_precision::Fp16 tiny_half = low_precision::fp16_from_float(0.00001f);
    expect_close(low_precision::fp16_to_float(tiny_half), 0.00001f, 1e-5f);
    const low_precision::Bf16 rounded_bfloat = low_precision::bf16_from_float(1.1f);
    expect_close(low_precision::bf16_to_float(rounded_bfloat), 1.1f, 0.01f);
}

void test_typed_quantization_matches_fp32_reference() {
    const std::vector<float> input{448.0f, 1.0f, -3.0f, 0.5f, 0.0f};
    std::vector<low_precision::Fp16> half_input;
    half_input.reserve(input.size());
    for (const float value : input) {
        half_input.push_back(low_precision::fp16_from_float(value));
    }

    const auto expected = low_precision::quantize_mxfp8(input, 1, input.size());
    const auto actual = low_precision::quantize_mxfp8(half_input, 1, input.size());
    assert(actual.values == expected.values);
    assert(actual.scales == expected.scales);

    const auto decoded_half = low_precision::dequantize_mxfp8_fp16(actual);
    const auto decoded_bfloat = low_precision::dequantize_mxfp8_bf16(actual);
    assert(decoded_half.size() == input.size());
    assert(decoded_bfloat.size() == input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float expected_value = low_precision::dequantize_mxfp8(expected)[index];
        expect_close(low_precision::fp16_to_float(decoded_half[index]), expected_value);
        expect_close(low_precision::bf16_to_float(decoded_bfloat[index]), expected_value);
    }

    const auto expected_nv = low_precision::quantize_nvfp4(input, 1, input.size());
    const auto actual_nv = low_precision::quantize_nvfp4(half_input, 1, input.size());
    assert(actual_nv.values == expected_nv.values);
    assert(actual_nv.block_scales == expected_nv.block_scales);
    assert(actual_nv.global_scale == expected_nv.global_scale);
    const auto nv_half = low_precision::dequantize_nvfp4_fp16(actual_nv);
    const auto nv_bfloat = low_precision::dequantize_nvfp4_bf16(actual_nv);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const float expected_value = low_precision::dequantize_nvfp4(expected_nv)[index];
        expect_close(low_precision::fp16_to_float(nv_half[index]), expected_value);
        expect_close(low_precision::bf16_to_float(nv_bfloat[index]), expected_value);
    }
}

}  // namespace

int main() {
    test_storage_round_trip();
    test_typed_quantization_matches_fp32_reference();
    std::cout << "FP16/BF16 reference tests passed\n";
}
