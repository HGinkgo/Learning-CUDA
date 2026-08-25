#include "low_precision/io.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

const char* kValidPath = "/tmp/low_precision_io_test.qnt";
const char* kCorruptPath = "/tmp/low_precision_io_test_corrupt.qnt";

void test_round_trip() {
    low_precision::QuantizedFile expected;
    expected.format = low_precision::Format::NvFp4;
    expected.input_type = low_precision::InputType::Fp16;
    expected.rounding = low_precision::Rounding::Stochastic;
    expected.rows = 3;
    expected.cols = 17;
    expected.block_size = 16;
    expected.scale_mode = low_precision::ScaleMode::TensorAndBlock;
    expected.global_scale = 0.125f;
    expected.values.resize(26);
    for (std::size_t index = 0; index < expected.values.size(); ++index) {
        expected.values[index] = static_cast<std::uint8_t>(index * 3u);
    }
    expected.scales = {0x38, 0x40, 0x42, 0x44, 0x46, 0x48};

    low_precision::write_quantized_file(kValidPath, expected);
    const auto actual = low_precision::read_quantized_file(kValidPath);
    assert(actual.format == expected.format);
    assert(actual.input_type == expected.input_type);
    assert(actual.rounding == expected.rounding);
    assert(actual.rows == expected.rows);
    assert(actual.cols == expected.cols);
    assert(actual.block_size == expected.block_size);
    assert(actual.scale_mode == expected.scale_mode);
    assert(actual.global_scale == expected.global_scale);
    assert(actual.values == expected.values);
    assert(actual.scales == expected.scales);
}

void test_rejects_invalid_magic_and_truncation() {
    {
        std::ofstream output(kCorruptPath, std::ios::binary);
        output << "bad";
    }
    bool rejected = false;
    try {
        static_cast<void>(low_precision::read_quantized_file(kCorruptPath));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);

    low_precision::QuantizedFile valid;
    valid.rows = 1;
    valid.cols = 1;
    valid.block_size = 32;
    valid.scale_mode = low_precision::ScaleMode::Block;
    valid.values = {0};
    valid.scales = {127};
    low_precision::write_quantized_file(kCorruptPath, valid);
    {
        std::fstream file(kCorruptPath, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(-1, std::ios::end);
        file.put('\0');
        file.close();
    }
    std::remove(kCorruptPath);
    std::ofstream truncated(kCorruptPath, std::ios::binary);
    truncated << "LPCQNT1";
    truncated.close();
    rejected = false;
    try {
        static_cast<void>(low_precision::read_quantized_file(kCorruptPath));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}

}  // namespace

int main() {
    test_round_trip();
    test_rejects_invalid_magic_and_truncation();
    std::remove(kValidPath);
    std::remove(kCorruptPath);
}
