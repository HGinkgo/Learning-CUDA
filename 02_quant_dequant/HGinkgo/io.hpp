#pragma once

#include "quant.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace low_precision {

enum class InputType : std::uint32_t { Fp32 = 1, Fp16 = 2 };

struct QuantizedFile {
    Format format = Format::MxFp8;
    InputType input_type = InputType::Fp32;
    Rounding rounding = Rounding::NearestEven;
    std::size_t rows = 0;
    std::size_t cols = 0;
    float global_scale = 1.0f;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
};

void write_quantized_file(const std::string& path, const QuantizedFile& file);
QuantizedFile read_quantized_file(const std::string& path);

}  // namespace low_precision
