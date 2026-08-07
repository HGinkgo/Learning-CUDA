// 作者：HGinkgo
// 测试环境：NVIDIA GeForce RTX 3090 (SM 8.6)，CUDA Toolkit 13.0
// 官方测试：54/54 通过（1 次预热，10 次性能采样）
// FlashAttention Case 13：float 12.510 ms，half 1.871 ms
// FlashAttention Case 14：float 61.226 ms，half 15.285 ms

#include <cuda_fp16.h>

#include <cfloat>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../tester/utils.h"

namespace {

constexpr int kRmsNormBlockSize = 256;
constexpr int kScalarAttentionBlockSize = 256;
constexpr int kMaximumHeadDim = 256;
constexpr int kShuffleMaskWidth = 32;
constexpr int kSubwarpSize = 16;
#if defined(PLATFORM_ILUVATAR)
constexpr int kSubwarpAttentionBlockSize = 1024;
#else
constexpr int kSubwarpAttentionBlockSize = 256;
#endif
constexpr int kSubwarpsPerBlock =
    kSubwarpAttentionBlockSize / kSubwarpSize;

size_t checkedElementCount(std::initializer_list<size_t> dimensions) {
  size_t count = 1;
  for (const size_t dimension : dimensions) {
    if (dimension != 0 &&
        count > std::numeric_limits<size_t>::max() / dimension) {
      throw std::overflow_error("tensor element count overflow");
    }
    count *= dimension;
  }
  return count;
}

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value) {
  return __float2half(value);
}

template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t hidden_dim, float eps) {
  __shared__ float partial_sums[kRmsNormBlockSize];

  const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_dim;
  float sum = 0.0f;
  for (size_t column = threadIdx.x; column < hidden_dim; column += blockDim.x) {
    const float value = toFloat(input[row_offset + column]);
    sum += value * value;
  }

  partial_sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(partial_sums[0] / static_cast<float>(hidden_dim) + eps);
  for (size_t column = threadIdx.x; column < hidden_dim; column += blockDim.x) {
    const float normalized = toFloat(input[row_offset + column]) * inverse_rms *
                             toFloat(weight[column]);
    output[row_offset + column] = fromFloat<T>(normalized);
  }
}

template <typename T, int static_head_dim>
__device__ __forceinline__ float scalarAttentionScore(const float* query,
                                                      const T* key,
                                                      size_t key_offset,
                                                      int head_dim,
                                                      float scale) {
  float score = 0.0f;
  if constexpr (static_head_dim == 0) {
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      score += query[dimension] * toFloat(key[key_offset + dimension]);
    }
  } else {
#pragma unroll
    for (int dimension = 0; dimension < static_head_dim; ++dimension) {
      score += query[dimension] * toFloat(key[key_offset + dimension]);
    }
  }
  return score * scale;
}

template <typename T, int static_head_dim = 0>
__global__ void flashAttentionScalarKernel(const T* query, const T* key,
                                           const T* value, T* output,
                                           int batch_size, int target_seq_len,
                                           int src_seq_len, int query_heads,
                                           int kv_heads, int head_dim,
                                           bool is_causal) {
  const size_t query_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t query_count =
      static_cast<size_t>(batch_size) * target_seq_len * query_heads;
  if (query_index >= query_count) {
    return;
  }

  size_t remaining = query_index;
  const int query_head = static_cast<int>(remaining % query_heads);
  remaining /= query_heads;
  const int target_position = static_cast<int>(remaining % target_seq_len);
  const int batch = static_cast<int>(remaining / target_seq_len);

  const int heads_per_kv = query_heads / kv_heads;
  const int kv_head = query_head / heads_per_kv;
  const size_t query_offset =
      ((static_cast<size_t>(batch) * target_seq_len + target_position) *
           query_heads +
       query_head) *
      head_dim;

  float query_cache[kMaximumHeadDim];
  float output_accumulator[kMaximumHeadDim];
  for (int dimension = 0; dimension < head_dim; ++dimension) {
    query_cache[dimension] = toFloat(query[query_offset + dimension]);
    output_accumulator[dimension] = 0.0f;
  }

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  float maximum_score = -FLT_MAX;
  for (int source_position = 0; source_position < src_seq_len;
       ++source_position) {
    if (is_causal && source_position > target_position) {
      continue;
    }
    const size_t key_offset =
        ((static_cast<size_t>(batch) * src_seq_len + source_position) *
             kv_heads +
         kv_head) *
        head_dim;
    maximum_score = fmaxf(maximum_score,
                          scalarAttentionScore<T, static_head_dim>(
                              query_cache, key, key_offset, head_dim, scale));
  }

  float softmax_denominator = 0.0f;
  for (int source_position = 0; source_position < src_seq_len;
       ++source_position) {
    if (is_causal && source_position > target_position) {
      continue;
    }
    const size_t key_offset =
        ((static_cast<size_t>(batch) * src_seq_len + source_position) *
             kv_heads +
         kv_head) *
        head_dim;
    softmax_denominator +=
        expf(scalarAttentionScore<T, static_head_dim>(
                 query_cache, key, key_offset, head_dim, scale) -
             maximum_score);
  }
  const float inverse_denominator =
      softmax_denominator == 0.0f ? 0.0f : 1.0f / softmax_denominator;

  for (int source_position = 0; source_position < src_seq_len;
       ++source_position) {
    if (is_causal && source_position > target_position) {
      continue;
    }
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + source_position) *
             kv_heads +
         kv_head) *
        head_dim;
    const float probability =
        expf(scalarAttentionScore<T, static_head_dim>(
                 query_cache, key, kv_offset, head_dim, scale) -
             maximum_score) *
        inverse_denominator;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      output_accumulator[dimension] +=
          probability * toFloat(value[kv_offset + dimension]);
    }
  }

  for (int dimension = 0; dimension < head_dim; ++dimension) {
    output[query_offset + dimension] =
        fromFloat<T>(output_accumulator[dimension]);
  }
}

#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
template <typename T, int static_head_dim>
__device__ __forceinline__ float subwarpAttentionScore(
    const float* query_values, const T* key, size_t key_offset, float scale,
    int lane_in_subwarp, unsigned int subwarp_mask) {
  constexpr int values_per_lane = static_head_dim / kSubwarpSize;
  float partial_score = 0.0f;
#if defined(PLATFORM_ILUVATAR)
  if constexpr (std::is_same_v<T, float>) {
    // Match the reference QK accumulation order for its strict float tolerance.
#pragma unroll
    for (int slot = 0; slot < values_per_lane; ++slot) {
#pragma unroll
      for (int source_lane = 0; source_lane < kSubwarpSize; ++source_lane) {
        const float query_value = __shfl_sync(
            subwarp_mask, query_values[slot], source_lane, kSubwarpSize);
        if (lane_in_subwarp == 0) {
          const int dimension = source_lane + slot * kSubwarpSize;
          partial_score += query_value * key[key_offset + dimension];
        }
      }
    }
    return __shfl_sync(subwarp_mask, partial_score, 0, kSubwarpSize) * scale;
  }
#endif
#pragma unroll
  for (int slot = 0; slot < values_per_lane; ++slot) {
    const int dimension = lane_in_subwarp + slot * kSubwarpSize;
    partial_score += query_values[slot] * toFloat(key[key_offset + dimension]);
  }
  for (int offset = kSubwarpSize / 2; offset > 0; offset /= 2) {
    partial_score += __shfl_down_sync(subwarp_mask, partial_score, offset,
                                      kSubwarpSize);
  }
  return __shfl_sync(subwarp_mask, partial_score, 0, kSubwarpSize) * scale;
}

template <typename T, int static_head_dim>
__global__ void flashAttentionSubwarpKernel(const T* query, const T* key,
                                            const T* value, T* output,
                                            int batch_size, int target_seq_len,
                                            int src_seq_len, int query_heads,
                                            int kv_heads, bool is_causal) {
  static_assert(static_head_dim % kSubwarpSize == 0 &&
                static_head_dim <= kMaximumHeadDim);
  // CoreX repeats its 32-bit shuffle mask across each half of a 64-lane warp.
  const int lane_in_half_warp = threadIdx.x % kShuffleMaskWidth;
  const int lane_in_subwarp = lane_in_half_warp % kSubwarpSize;
  const unsigned int subwarp_mask =
      lane_in_half_warp < kSubwarpSize ? 0x0000ffffu : 0xffff0000u;
  const int subwarp_index_in_block = threadIdx.x / kSubwarpSize;
  const size_t query_index =
      static_cast<size_t>(blockIdx.x) * kSubwarpsPerBlock +
      subwarp_index_in_block;
  const size_t query_count =
      static_cast<size_t>(batch_size) * target_seq_len * query_heads;
  if (query_index >= query_count) {
    return;
  }

  size_t remaining = query_index;
  const int query_head = static_cast<int>(remaining % query_heads);
  remaining /= query_heads;
  const int target_position = static_cast<int>(remaining % target_seq_len);
  const int batch = static_cast<int>(remaining / target_seq_len);
  const int heads_per_kv = query_heads / kv_heads;
  const int kv_head = query_head / heads_per_kv;

  const size_t query_offset =
      ((static_cast<size_t>(batch) * target_seq_len + target_position) *
           query_heads +
       query_head) *
      static_head_dim;

  constexpr int values_per_lane = static_head_dim / kSubwarpSize;
  float query_values[values_per_lane];
  float output_accumulator[values_per_lane];
#pragma unroll
  for (int slot = 0; slot < values_per_lane; ++slot) {
    const int dimension = lane_in_subwarp + slot * kSubwarpSize;
    query_values[slot] = toFloat(query[query_offset + dimension]);
    output_accumulator[slot] = 0.0f;
  }

  const float scale = 1.0f / sqrtf(static_cast<float>(static_head_dim));
  float maximum_score = -FLT_MAX;
  float softmax_denominator = 0.0f;
  for (int source_position = 0; source_position < src_seq_len;
       ++source_position) {
    if (is_causal && source_position > target_position) {
      continue;
    }
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + source_position) *
             kv_heads +
         kv_head) *
        static_head_dim;
    const float score = subwarpAttentionScore<T, static_head_dim>(
        query_values, key, kv_offset, scale, lane_in_subwarp, subwarp_mask);
    float previous_scale = 0.0f;
    float score_scale = 0.0f;
    if (lane_in_subwarp == 0) {
      const float next_maximum_score = fmaxf(maximum_score, score);
      previous_scale = expf(maximum_score - next_maximum_score);
      score_scale = expf(score - next_maximum_score);
      softmax_denominator = softmax_denominator * previous_scale + score_scale;
      maximum_score = next_maximum_score;
    }
    previous_scale =
        __shfl_sync(subwarp_mask, previous_scale, 0, kSubwarpSize);
    score_scale = __shfl_sync(subwarp_mask, score_scale, 0, kSubwarpSize);

#pragma unroll
    for (int slot = 0; slot < values_per_lane; ++slot) {
      const int dimension = lane_in_subwarp + slot * kSubwarpSize;
      output_accumulator[slot] =
          output_accumulator[slot] * previous_scale +
          score_scale * toFloat(value[kv_offset + dimension]);
    }
  }

  float inverse_denominator = 0.0f;
  if (lane_in_subwarp == 0) {
    inverse_denominator =
        softmax_denominator == 0.0f ? 0.0f : 1.0f / softmax_denominator;
  }
  inverse_denominator =
      __shfl_sync(subwarp_mask, inverse_denominator, 0, kSubwarpSize);
#pragma unroll
  for (int slot = 0; slot < values_per_lane; ++slot) {
    const int dimension = lane_in_subwarp + slot * kSubwarpSize;
    output[query_offset + dimension] =
        fromFloat<T>(output_accumulator[slot] * inverse_denominator);
  }
}
#endif

}  // namespace

#if defined(PLATFORM_NVIDIA) && CUDART_VERSION >= 13000
// The supplied tester was built with CUDA 12 and references this ABI symbol,
// which CUDA 13 no longer exports. The public API has the same semantics.
extern "C" cudaError_t cudaGetDeviceProperties_v2(cudaDeviceProp* prop,
                                                  int device) {
  return cudaGetDeviceProperties(prop, device);
}
#endif

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  if (hidden_dim == 0) {
    throw std::invalid_argument("hidden_dim must be positive");
  }
  const size_t input_elements = checkedElementCount({rows, hidden_dim});
  if (h_input.size() != input_elements || h_weight.size() != hidden_dim) {
    throw std::invalid_argument("RMSNorm input size mismatch");
  }
  if (rows == 0) {
    h_output.clear();
    return;
  }

  const size_t input_bytes = checkedElementCount({input_elements, sizeof(T)});
  const size_t weight_bytes = checkedElementCount({hidden_dim, sizeof(T)});
  h_output.resize(input_elements);

  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;
  RUNTIME_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_input), input_bytes));
  RUNTIME_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_weight), weight_bytes));
  RUNTIME_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_output), input_bytes));

  RUNTIME_CHECK(
      cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                           cudaMemcpyHostToDevice));

  rmsNormKernel<<<rows, kRmsNormBlockSize>>>(d_input, d_weight, d_output,
                                            hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, input_bytes,
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_output));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_input));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float or half) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads,
 * head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads,
 * head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len,
 * query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query
 * attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim,
                    bool is_causal) {
  if (batch_size < 0 || target_seq_len < 0 || src_seq_len <= 0) {
    throw std::invalid_argument(
        "attention dimensions must be non-negative and source non-empty");
  }
  if (query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0) {
    throw std::invalid_argument(
        "query_heads must be divisible by positive kv_heads");
  }
  if (head_dim <= 0 || head_dim > kMaximumHeadDim) {
    throw std::invalid_argument("head_dim must be in [1, 256]");
  }

  const size_t query_count = checkedElementCount(
      {static_cast<size_t>(batch_size), static_cast<size_t>(target_seq_len),
       static_cast<size_t>(query_heads)});
  const size_t query_elements =
      checkedElementCount({query_count, static_cast<size_t>(head_dim)});
  const size_t kv_elements = checkedElementCount(
      {static_cast<size_t>(batch_size), static_cast<size_t>(src_seq_len),
       static_cast<size_t>(kv_heads), static_cast<size_t>(head_dim)});
  if (h_q.size() != query_elements || h_k.size() != kv_elements ||
      h_v.size() != kv_elements) {
    throw std::invalid_argument("attention input size mismatch");
  }
  if (query_count == 0) {
    h_o.clear();
    return;
  }
#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
#if defined(PLATFORM_ILUVATAR)
  constexpr bool supports_subwarp_type = true;
#else
  constexpr bool supports_subwarp_type = std::is_same_v<T, half>;
#endif
  const bool uses_subwarp =
      supports_subwarp_type && head_dim >= kSubwarpSize &&
      (head_dim & (head_dim - 1)) == 0;
  const size_t queries_per_block =
      uses_subwarp ? kSubwarpsPerBlock : kScalarAttentionBlockSize;
#else
  const size_t queries_per_block = kScalarAttentionBlockSize;
#endif
  const size_t block_count =
      (query_count + queries_per_block - 1) / queries_per_block;
  if (block_count > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("attention grid is too large");
  }
  const unsigned int grid_blocks = static_cast<unsigned int>(block_count);

  const size_t query_bytes = checkedElementCount({query_elements, sizeof(T)});
  const size_t kv_bytes = checkedElementCount({kv_elements, sizeof(T)});
  h_o.resize(query_elements);

  T* d_query = nullptr;
  T* d_key = nullptr;
  T* d_value = nullptr;
  T* d_output = nullptr;
  RUNTIME_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_query), query_bytes));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_key), kv_bytes));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_value), kv_bytes));
  RUNTIME_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_output), query_bytes));

  RUNTIME_CHECK(
      cudaMemcpy(d_query, h_q.data(), query_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(
      cudaMemcpy(d_key, h_k.data(), kv_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(
      cudaMemcpy(d_value, h_v.data(), kv_bytes, cudaMemcpyHostToDevice));

#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
  if (uses_subwarp) {
    switch (head_dim) {
      case 16:
        flashAttentionSubwarpKernel<T, 16>
            <<<grid_blocks, kSubwarpAttentionBlockSize>>>(
            d_query, d_key, d_value, d_output, batch_size, target_seq_len,
            src_seq_len, query_heads, kv_heads, is_causal);
        break;
      case 32:
        flashAttentionSubwarpKernel<T, 32>
            <<<grid_blocks, kSubwarpAttentionBlockSize>>>(
            d_query, d_key, d_value, d_output, batch_size, target_seq_len,
            src_seq_len, query_heads, kv_heads, is_causal);
        break;
      case 64:
        flashAttentionSubwarpKernel<T, 64>
            <<<grid_blocks, kSubwarpAttentionBlockSize>>>(
            d_query, d_key, d_value, d_output, batch_size, target_seq_len,
            src_seq_len, query_heads, kv_heads, is_causal);
        break;
      case 128:
        flashAttentionSubwarpKernel<T, 128>
            <<<grid_blocks, kSubwarpAttentionBlockSize>>>(
            d_query, d_key, d_value, d_output, batch_size, target_seq_len,
            src_seq_len, query_heads, kv_heads, is_causal);
        break;
      case 256:
        flashAttentionSubwarpKernel<T, 256>
            <<<grid_blocks, kSubwarpAttentionBlockSize>>>(
            d_query, d_key, d_value, d_output, batch_size, target_seq_len,
            src_seq_len, query_heads, kv_heads, is_causal);
        break;
    }
  } else {
#endif
    if (head_dim == 32) {
      flashAttentionScalarKernel<T, 32>
          <<<grid_blocks, kScalarAttentionBlockSize>>>(
          d_query, d_key, d_value, d_output, batch_size, target_seq_len,
          src_seq_len, query_heads, kv_heads, head_dim, is_causal);
    } else if (head_dim == 64) {
      flashAttentionScalarKernel<T, 64>
          <<<grid_blocks, kScalarAttentionBlockSize>>>(
          d_query, d_key, d_value, d_output, batch_size, target_seq_len,
          src_seq_len, query_heads, kv_heads, head_dim, is_causal);
    } else {
      flashAttentionScalarKernel<T>
          <<<grid_blocks, kScalarAttentionBlockSize>>>(
          d_query, d_key, d_value, d_output, batch_size, target_seq_len,
          src_seq_len, query_heads, kv_heads, head_dim, is_causal);
    }
#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
  }
#endif
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(
      cudaMemcpy(h_o.data(), d_output, query_bytes, cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_output));
  RUNTIME_CHECK(cudaFree(d_value));
  RUNTIME_CHECK(cudaFree(d_key));
  RUNTIME_CHECK(cudaFree(d_query));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&,
                             const std::vector<float>&, std::vector<float>&,
                             size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
                            std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&,
                                    const std::vector<float>&,
                                    const std::vector<float>&,
                                    std::vector<float>&, int, int, int, int,
                                    int, int, bool);
template void flashAttention<half>(const std::vector<half>&,
                                   const std::vector<half>&,
                                   const std::vector<half>&, std::vector<half>&,
                                   int, int, int, int, int, int, bool);
