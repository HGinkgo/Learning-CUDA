// Author: HGinkgo

#include <musa_fp16.h>

#include <cfloat>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../tester/utils.h"

namespace {

const int kRmsNormBlockSize = 256;
const int kAttentionBlockSize = 256;
const int kMaximumHeadDim = 256;

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

__device__ __forceinline__ size_t headOffset(
    int batch, int position, int sequence_length, int heads, int head,
    int head_dim) {
  return ((static_cast<size_t>(batch) * sequence_length + position) * heads +
          head) *
         head_dim;
}

template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t hidden_dim, float eps) {
  __shared__ float partial_sums[kRmsNormBlockSize];

  const size_t row_offset = static_cast<size_t>(blockIdx.x) * hidden_dim;
  float sum = 0.0f;
  for (size_t column = threadIdx.x; column < hidden_dim;
       column += blockDim.x) {
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
  for (size_t column = threadIdx.x; column < hidden_dim;
       column += blockDim.x) {
    const float normalized = toFloat(input[row_offset + column]) *
                             inverse_rms * toFloat(weight[column]);
    output[row_offset + column] = fromFloat<T>(normalized);
  }
}

template <typename T, int static_head_dim>
__device__ __forceinline__ float attentionScore(const float* query,
                                                const T* key,
                                                size_t key_offset,
                                                int head_dim, float scale) {
  float score = 0.0f;
  const int dimension_count =
      static_head_dim == 0 ? head_dim : static_head_dim;
#pragma unroll
  for (int dimension = 0; dimension < dimension_count; ++dimension) {
    score += query[dimension] * toFloat(key[key_offset + dimension]);
  }
  return score * scale;
}

template <typename T, int static_head_dim>
__global__ void scalarAttentionKernel(
    const T* query, const T* key, const T* value, T* output, int batch_size,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
    int head_dim, bool is_causal) {
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
  const int kv_head = query_head / (query_heads / kv_heads);
  const size_t query_offset = headOffset(
      batch, target_position, target_seq_len, query_heads, query_head,
      head_dim);

  const int dimension_count =
      static_head_dim == 0 ? head_dim : static_head_dim;
  float query_cache[static_head_dim == 0 ? kMaximumHeadDim : static_head_dim];
  float output_accumulator[static_head_dim == 0 ? kMaximumHeadDim
                                                : static_head_dim];
#pragma unroll
  for (int dimension = 0; dimension < dimension_count; ++dimension) {
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
    const size_t key_offset = headOffset(
        batch, source_position, src_seq_len, kv_heads, kv_head, head_dim);
    maximum_score =
        fmaxf(maximum_score,
              attentionScore<T, static_head_dim>(
                  query_cache, key, key_offset, head_dim, scale));
  }

  float denominator = 0.0f;
  for (int source_position = 0; source_position < src_seq_len;
       ++source_position) {
    if (is_causal && source_position > target_position) {
      continue;
    }
    const size_t key_offset = headOffset(
        batch, source_position, src_seq_len, kv_heads, kv_head, head_dim);
    denominator +=
        expf(attentionScore<T, static_head_dim>(
                 query_cache, key, key_offset, head_dim, scale) -
             maximum_score);
  }
  const float inverse_denominator =
      denominator == 0.0f ? 0.0f : 1.0f / denominator;

  for (int source_position = 0; source_position < src_seq_len;
       ++source_position) {
    if (is_causal && source_position > target_position) {
      continue;
    }
    const size_t kv_offset = headOffset(
        batch, source_position, src_seq_len, kv_heads, kv_head, head_dim);
    const float probability =
        expf(attentionScore<T, static_head_dim>(
                 query_cache, key, kv_offset, head_dim, scale) -
             maximum_score) *
        inverse_denominator;
#pragma unroll
    for (int dimension = 0; dimension < dimension_count; ++dimension) {
      output_accumulator[dimension] +=
          probability * toFloat(value[kv_offset + dimension]);
    }
  }

#pragma unroll
  for (int dimension = 0; dimension < dimension_count; ++dimension) {
    output[query_offset + dimension] =
        fromFloat<T>(output_accumulator[dimension]);
  }
}

template <int head_dim, int subgroup_size, typename T>
__device__ __forceinline__ void loadSourceQuery(
    const T* query, size_t query_offset, int lane, float* query_values) {
  const int values_per_lane = head_dim / subgroup_size;
#pragma unroll
  for (int slot = 0; slot < values_per_lane; ++slot) {
    const int dimension = lane + slot * subgroup_size;
    query_values[slot] = toFloat(query[query_offset + dimension]);
  }
}

template <int head_dim, int subgroup_size>
__device__ __forceinline__ void loadSourceQuery(
    const half* query, size_t query_offset, int lane, float* query_values) {
  const int pairs_per_lane = head_dim / (2 * subgroup_size);
  const __half2* packed_query =
      reinterpret_cast<const __half2*>(query + query_offset);
#pragma unroll
  for (int pair = 0; pair < pairs_per_lane; ++pair) {
    const float2 values = __half22float2(
        packed_query[lane + pair * subgroup_size]);
    query_values[pair * 2] = values.x;
    query_values[pair * 2 + 1] = values.y;
  }
}

template <int head_dim, int subgroup_size, typename T>
__device__ __forceinline__ float orderedSourceScore(
    const float* query_values, const T* key, size_t key_offset, float scale,
    unsigned int subgroup_mask) {
  const int values_per_lane = head_dim / subgroup_size;
  float score = 0.0f;
#pragma unroll
  for (int slot = 0; slot < values_per_lane; ++slot) {
#pragma unroll
    for (int query_lane = 0; query_lane < subgroup_size;
         ++query_lane) {
      const int dimension = slot * subgroup_size + query_lane;
      const float query_value = __shfl_sync(
          subgroup_mask, query_values[slot], query_lane,
          subgroup_size);
      score += query_value * toFloat(key[key_offset + dimension]);
    }
  }
  return score * scale;
}

template <int head_dim, int subgroup_size>
__device__ __forceinline__ float orderedSourceScore(
    const float* query_values, const half* key, size_t key_offset, float scale,
    unsigned int subgroup_mask) {
  float score = 0.0f;
  const int pairs_per_lane = head_dim / (2 * subgroup_size);
  const __half2* packed_key =
      reinterpret_cast<const __half2*>(key + key_offset);
#pragma unroll
  for (int pair = 0; pair < pairs_per_lane; ++pair) {
#pragma unroll
    for (int query_lane = 0; query_lane < subgroup_size;
         ++query_lane) {
      const float query_low = __shfl_sync(
          subgroup_mask, query_values[pair * 2], query_lane,
          subgroup_size);
      const float query_high = __shfl_sync(
          subgroup_mask, query_values[pair * 2 + 1], query_lane,
          subgroup_size);
      const float2 key_values = __half22float2(
          packed_key[query_lane + pair * subgroup_size]);
      score += query_low * key_values.x;
      score += query_high * key_values.y;
    }
  }
  return score * scale;
}

template <int head_dim, int subgroup_size, typename T>
__device__ __forceinline__ void updateSourceOutput(
    const T* value, size_t value_offset, int lane, float previous_scale,
    float score_scale, float* output_accumulator) {
  const int values_per_lane = head_dim / subgroup_size;
#pragma unroll
  for (int slot = 0; slot < values_per_lane; ++slot) {
    const int dimension = lane + slot * subgroup_size;
    output_accumulator[slot] =
        output_accumulator[slot] * previous_scale +
        score_scale * toFloat(value[value_offset + dimension]);
  }
}

template <int head_dim, int subgroup_size>
__device__ __forceinline__ void updateSourceOutput(
    const half* value, size_t value_offset, int lane, float previous_scale,
    float score_scale, float* output_accumulator) {
  const int pairs_per_lane = head_dim / (2 * subgroup_size);
  const __half2* packed_value =
      reinterpret_cast<const __half2*>(value + value_offset);
#pragma unroll
  for (int pair = 0; pair < pairs_per_lane; ++pair) {
    const float2 values = __half22float2(
        packed_value[lane + pair * subgroup_size]);
    output_accumulator[pair * 2] =
        output_accumulator[pair * 2] * previous_scale +
        score_scale * values.x;
    output_accumulator[pair * 2 + 1] =
        output_accumulator[pair * 2 + 1] * previous_scale +
        score_scale * values.y;
  }
}

template <int head_dim, int subgroup_size, typename T>
__device__ __forceinline__ void storeSourceOutput(
    T* output, size_t query_offset, int lane, float inverse_denominator,
    const float* output_accumulator) {
  const int values_per_lane = head_dim / subgroup_size;
#pragma unroll
  for (int slot = 0; slot < values_per_lane; ++slot) {
    const int dimension = lane + slot * subgroup_size;
    output[query_offset + dimension] =
        fromFloat<T>(output_accumulator[slot] * inverse_denominator);
  }
}

template <int head_dim, int subgroup_size>
__device__ __forceinline__ void storeSourceOutput(
    half* output, size_t query_offset, int lane, float inverse_denominator,
    const float* output_accumulator) {
  const int pairs_per_lane = head_dim / (2 * subgroup_size);
  __half2* packed_output =
      reinterpret_cast<__half2*>(output + query_offset);
#pragma unroll
  for (int pair = 0; pair < pairs_per_lane; ++pair) {
    packed_output[lane + pair * subgroup_size] =
        __floats2half2_rn(
            output_accumulator[pair * 2] * inverse_denominator,
            output_accumulator[pair * 2 + 1] * inverse_denominator);
  }
}

template <typename T, int head_dim, int subgroup_size>
__global__ void sourceParallelAttentionKernel(
    const T* query, const T* key, const T* value, T* output, int batch_size,
    int target_seq_len, int src_seq_len, int query_heads, int kv_heads,
    bool is_causal) {
  static_assert(head_dim % subgroup_size == 0 &&
                    head_dim <= kMaximumHeadDim,
                "unsupported subgroup dimensions");
  static_assert(!std::is_same<T, half>::value ||
                    head_dim >= 2 * subgroup_size,
                "half subgroup requires aligned half2 pairs");
  // Lanes compute consecutive source scores, then replay them in order for
  // online softmax while each lane owns a slice of the output dimensions.
  const int lane_in_warp = threadIdx.x % 32;
  const int lane = lane_in_warp % subgroup_size;
  const unsigned int subgroup_mask =
      ((1u << subgroup_size) - 1u)
      << ((lane_in_warp / subgroup_size) * subgroup_size);
  const int subgroup_index = threadIdx.x / subgroup_size;
  const int subgroups_per_block = kAttentionBlockSize / subgroup_size;
  const size_t query_index =
      static_cast<size_t>(blockIdx.x) * subgroups_per_block + subgroup_index;
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
  const int kv_head = query_head / (query_heads / kv_heads);
  const size_t query_offset = headOffset(
      batch, target_position, target_seq_len, query_heads, query_head,
      head_dim);

  const int values_per_lane = head_dim / subgroup_size;
  float query_values[values_per_lane];
  float output_accumulator[values_per_lane];
  loadSourceQuery<head_dim, subgroup_size>(query, query_offset, lane,
                                           query_values);
#pragma unroll
  for (int slot = 0; slot < values_per_lane; ++slot) {
    output_accumulator[slot] = 0.0f;
  }

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  float maximum_score = -FLT_MAX;
  float denominator = 0.0f;
  const int source_count =
      is_causal && target_position < src_seq_len ? target_position + 1
                                                 : src_seq_len;
  for (int source_base = 0; source_base < source_count;
       source_base += subgroup_size) {
    int local_source = source_base + lane;
    if (local_source >= source_count) {
      local_source = source_count - 1;
    }
    const size_t local_key_offset = headOffset(
        batch, local_source, src_seq_len, kv_heads, kv_head, head_dim);
    const float local_score = orderedSourceScore<head_dim, subgroup_size>(
        query_values, key, local_key_offset, scale, subgroup_mask);
    const int tile_size =
        source_count - source_base < subgroup_size
            ? source_count - source_base
            : subgroup_size;

    for (int source_slot = 0; source_slot < tile_size; ++source_slot) {
      const int source_position = source_base + source_slot;
      const float score = __shfl_sync(subgroup_mask, local_score, source_slot,
                                      subgroup_size);
      const size_t value_offset = headOffset(
          batch, source_position, src_seq_len, kv_heads, kv_head, head_dim);

      float previous_scale = 0.0f;
      float score_scale = 0.0f;
      if (lane == 0) {
        const float next_maximum = fmaxf(maximum_score, score);
        previous_scale = expf(maximum_score - next_maximum);
        score_scale = expf(score - next_maximum);
        denominator = denominator * previous_scale + score_scale;
        maximum_score = next_maximum;
      }
      previous_scale = __shfl_sync(subgroup_mask, previous_scale, 0,
                                   subgroup_size);
      score_scale = __shfl_sync(subgroup_mask, score_scale, 0,
                                subgroup_size);

      updateSourceOutput<head_dim, subgroup_size>(
          value, value_offset, lane, previous_scale, score_scale,
          output_accumulator);
    }
  }

  float inverse_denominator = 0.0f;
  if (lane == 0) {
    inverse_denominator = denominator == 0.0f ? 0.0f : 1.0f / denominator;
  }
  inverse_denominator = __shfl_sync(subgroup_mask, inverse_denominator, 0,
                                    subgroup_size);
  storeSourceOutput<head_dim, subgroup_size>(
      output, query_offset, lane, inverse_denominator, output_accumulator);
}

}  // namespace

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  if (hidden_dim == 0) {
    throw std::invalid_argument("hidden_dim must be positive");
  }
  const size_t element_count = checkedElementCount({rows, hidden_dim});
  if (h_input.size() != element_count || h_weight.size() != hidden_dim) {
    throw std::invalid_argument("RMSNorm input size mismatch");
  }
  if (rows == 0) {
    h_output.clear();
    return;
  }
  if (rows > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("RMSNorm grid is too large");
  }

  const size_t input_bytes = checkedElementCount({element_count, sizeof(T)});
  const size_t weight_bytes = checkedElementCount({hidden_dim, sizeof(T)});
  h_output.resize(element_count);

  T* d_input = NULL;
  T* d_weight = NULL;
  T* d_output = NULL;
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_input), input_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_weight), weight_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_output), input_bytes));
  RUNTIME_CHECK(
      musaMemcpy(d_input, h_input.data(), input_bytes, musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_weight, h_weight.data(), weight_bytes,
                           musaMemcpyHostToDevice));

  rmsNormKernel<<<static_cast<unsigned int>(rows), kRmsNormBlockSize>>>(
      d_input, d_weight, d_output, hidden_dim, eps);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaMemcpy(h_output.data(), d_output, input_bytes,
                           musaMemcpyDeviceToHost));

  RUNTIME_CHECK(musaFree(d_output));
  RUNTIME_CHECK(musaFree(d_weight));
  RUNTIME_CHECK(musaFree(d_input));
}

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
      {static_cast<size_t>(batch_size),
       static_cast<size_t>(target_seq_len),
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

  const bool use_subgroups =
      head_dim >= 16 &&
      (head_dim & (head_dim - 1)) == 0 &&
      (!std::is_same<T, float>::value || head_dim >= 64);
  const int subgroup_size =
      std::is_same<T, half>::value && head_dim <= 64 ? 8 : 16;
  const size_t queries_per_block =
      use_subgroups ? kAttentionBlockSize / subgroup_size
                    : kAttentionBlockSize;
  const size_t block_count =
      (query_count + queries_per_block - 1) / queries_per_block;
  if (block_count > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("attention grid is too large");
  }
  const size_t query_bytes = checkedElementCount({query_elements, sizeof(T)});
  const size_t kv_bytes = checkedElementCount({kv_elements, sizeof(T)});
  h_o.resize(query_elements);

  T* d_query = NULL;
  T* d_key = NULL;
  T* d_value = NULL;
  T* d_output = NULL;
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_query), query_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_key), kv_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_value), kv_bytes));
  RUNTIME_CHECK(musaMalloc(reinterpret_cast<void**>(&d_output), query_bytes));
  RUNTIME_CHECK(
      musaMemcpy(d_query, h_q.data(), query_bytes, musaMemcpyHostToDevice));
  RUNTIME_CHECK(
      musaMemcpy(d_key, h_k.data(), kv_bytes, musaMemcpyHostToDevice));
  RUNTIME_CHECK(
      musaMemcpy(d_value, h_v.data(), kv_bytes, musaMemcpyHostToDevice));

  if (use_subgroups) {
#define LAUNCH_SOURCE_PARALLEL(DIM)                                          \
  sourceParallelAttentionKernel<                                            \
      T, DIM, (std::is_same<T, half>::value && DIM <= 64 ? 8 : 16)>        \
      <<<static_cast<unsigned int>(block_count), kAttentionBlockSize>>>(    \
          d_query, d_key, d_value, d_output, batch_size, target_seq_len,     \
          src_seq_len, query_heads, kv_heads, is_causal)
#define SOURCE_CASE(DIM)         \
  case DIM:                      \
    LAUNCH_SOURCE_PARALLEL(DIM); \
    break
    switch (head_dim) {
      SOURCE_CASE(16);
      SOURCE_CASE(32);
      SOURCE_CASE(64);
      SOURCE_CASE(128);
      SOURCE_CASE(256);
    }
#undef SOURCE_CASE
#undef LAUNCH_SOURCE_PARALLEL
  } else {
#define LAUNCH_SCALAR(DIM)                                                   \
  scalarAttentionKernel<T, DIM>                                             \
      <<<static_cast<unsigned int>(block_count), kAttentionBlockSize>>>(    \
          d_query, d_key, d_value, d_output, batch_size, target_seq_len,     \
          src_seq_len, query_heads, kv_heads, head_dim, is_causal)
#define SCALAR_CASE(DIM) \
  case DIM:              \
    LAUNCH_SCALAR(DIM);  \
    break
    switch (head_dim) {
      SCALAR_CASE(1);
      SCALAR_CASE(2);
      SCALAR_CASE(4);
      SCALAR_CASE(8);
      SCALAR_CASE(16);
      SCALAR_CASE(32);
      default:
        LAUNCH_SCALAR(0);
        break;
    }
#undef SCALAR_CASE
#undef LAUNCH_SCALAR
  }
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaMemcpy(h_o.data(), d_output, query_bytes,
                           musaMemcpyDeviceToHost));

  RUNTIME_CHECK(musaFree(d_output));
  RUNTIME_CHECK(musaFree(d_value));
  RUNTIME_CHECK(musaFree(d_key));
  RUNTIME_CHECK(musaFree(d_query));
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
