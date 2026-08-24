#include "quant.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using low_precision::Format;
using low_precision::Rounding;

namespace {

void expect_close(float actual, float expected, float tolerance = 1e-6f) {
    assert(std::fabs(actual - expected) <= tolerance);
}

void test_codebooks() {
    expect_close(low_precision::decode_e2m1(0), 0.0f);
    expect_close(low_precision::decode_e2m1(1), 0.5f);
    expect_close(low_precision::decode_e2m1(2), 1.0f);
    expect_close(low_precision::decode_e2m1(7), 6.0f);
    expect_close(low_precision::decode_e2m1(8), -0.0f);
    expect_close(low_precision::decode_e2m1(15), -6.0f);

    expect_close(low_precision::decode_e4m3(0x00), 0.0f);
    expect_close(low_precision::decode_e4m3(0x38), 1.0f);
    expect_close(low_precision::decode_e4m3(0x7e), 448.0f);
    expect_close(low_precision::decode_e4m3(0xfe), -448.0f);
}

void test_mxfp8_block_and_tail() {
    std::vector<float> input(35, 0.0f);
    input[0] = 448.0f;
    input[1] = 1.0f;
    input[32] = -3.0f;
    input[34] = 0.5f;

    const auto encoded = low_precision::quantize_mxfp8(
        input, 1, input.size(), Rounding::NearestEven);
    assert(encoded.values.size() == input.size());
    assert(encoded.scales.size() == 2);

    const auto decoded = low_precision::dequantize_mxfp8(encoded);
    assert(decoded.size() == input.size());
    expect_close(decoded[0], 448.0f);
    expect_close(decoded[32], -3.0f);
    expect_close(decoded[34], 0.5f);
}

void test_multiple_rows_and_tiny_nvfp4_block() {
    std::vector<float> mxfp8_input(2 * 33, 0.0f);
    mxfp8_input[0] = 448.0f;
    mxfp8_input[33] = -448.0f;
    const auto mxfp8 = low_precision::quantize_mxfp8(
        mxfp8_input, 2, 33, Rounding::NearestEven);
    assert(mxfp8.scales.size() == 4);
    const auto mxfp8_decoded = low_precision::dequantize_mxfp8(mxfp8);
    expect_close(mxfp8_decoded[0], 448.0f);
    expect_close(mxfp8_decoded[33], -448.0f);

    std::vector<float> nvfp4_input(2 * 17, 0.0f);
    nvfp4_input[0] = 1.0e-6f;
    nvfp4_input[17] = -1.0e-6f;
    const auto nvfp4 = low_precision::quantize_nvfp4(
        nvfp4_input, 2, 17, Rounding::NearestEven);
    assert(nvfp4.values.size() == 17);
    assert(nvfp4.block_scales.size() == 4);
    assert(nvfp4.block_scales[0] != 0);
    assert(nvfp4.block_scales[2] != 0);
}

void test_nvfp4_packing_and_tail() {
    std::vector<float> input{0.0f, 6.0f, -6.0f, 1.0f, 0.5f};
    const auto encoded = low_precision::quantize_nvfp4(
        input, 1, input.size(), Rounding::NearestEven);
    assert(encoded.values.size() == 3);
    assert(encoded.block_scales.size() == 1);
    assert((encoded.values[0] & 0x0f) == 0);
    assert((encoded.values[0] >> 4) == 0x07);
    assert((encoded.values[1] & 0x0f) == 0x0f);
    assert((encoded.values[1] >> 4) == 0x02);
    assert((encoded.values[2] & 0x0f) == 0x01);
    assert((encoded.values[2] >> 4) == 0);

    const auto decoded = low_precision::dequantize_nvfp4(encoded);
    assert(decoded.size() == input.size());
    expect_close(decoded[0], 0.0f);
    expect_close(decoded[1], 6.0f);
    expect_close(decoded[2], -6.0f);
    expect_close(decoded[3], 1.0f);
    expect_close(decoded[4], 0.5f);
}

}  // namespace

int main() {
    test_codebooks();
    test_mxfp8_block_and_tail();
    test_nvfp4_packing_and_tail();
    test_multiple_rows_and_tiny_nvfp4_block();
    std::cout << "CPU reference tests passed\n";
}
