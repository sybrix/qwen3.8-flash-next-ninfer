// Implements: include/ninfer/ops/ple.h
// Elementwise and per-(stream,column) reductions; the dilated convolution is parallel over
// (channel, column, row) because it reads its normalized input, not its own output.
#include "ops/launcher/ple.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/ple.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 256;
constexpr int kMaxGrid = 8192;

int elementwise_grid(std::int64_t n) {
    return static_cast<int>(std::clamp<std::int64_t>((n + kBlock - 1) / kBlock, 1, kMaxGrid));
}

__global__ void ple_dequantize_kernel(const __nv_fp8_e4m3* __restrict__ codes,
                                      const __nv_bfloat16* __restrict__ scale,
                                      __nv_bfloat16* __restrict__ out, std::int64_t n) {
    const float s = __bfloat162float(scale[0]);
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        out[i] = __float2bfloat16_rn(static_cast<float>(codes[i]) * s);
    }
}

// One CTA per (stream, column).
__global__ void ple_gate_kernel(const __nv_bfloat16* __restrict__ key,
                                const __nv_bfloat16* __restrict__ query,
                                const __nv_bfloat16* __restrict__ value,
                                __nv_bfloat16* __restrict__ out, std::int32_t h,
                                std::int32_t streams) {
    const std::int32_t s     = blockIdx.x % streams;
    const std::int64_t t     = blockIdx.x / streams;
    const std::int64_t base  = (t * streams + s) * static_cast<std::int64_t>(h);
    const __nv_bfloat16* v   = value + t * h;
    float dot                = 0.0f;
    for (std::int32_t i = threadIdx.x; i < h; i += blockDim.x) {
        dot += __bfloat162float(key[base + i]) * __bfloat162float(query[base + i]);
    }
    __shared__ float partial[kBlock / 32];
    dot = warp_reduce_sum(dot);
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = dot; }
    __syncthreads();
    if (threadIdx.x < 32) {
        float total = threadIdx.x < kBlock / 32 ? partial[threadIdx.x] : 0.0f;
        total       = warp_reduce_sum(total);
        if (threadIdx.x == 0) { partial[0] = total; }
    }
    __syncthreads();
    const float gamma  = partial[0] * rsqrtf(static_cast<float>(h));
    const float signed_root = copysignf(sqrtf(fmaxf(fabsf(gamma), 1.0e-6f)), gamma);
    const float gate   = 1.0f / (1.0f + __expf(-signed_root));
    for (std::int32_t i = threadIdx.x; i < h; i += blockDim.x) {
        out[base + i] = __float2bfloat16_rn(gate * __bfloat162float(v[i]));
    }
}

// Column j of row b reads u[c, j - 9 .. j]: negative offsets come from the row's initial window.
// state layout per slot: [C,9], element (c, k) at k*C + c, oldest first.
__global__ void ple_conv_kernel(const __nv_bfloat16* __restrict__ gated,
                                const __nv_bfloat16* __restrict__ normalized,
                                const __nv_bfloat16* __restrict__ weight,
                                const __nv_bfloat16* __restrict__ states,
                                const std::int32_t* __restrict__ initial_slots,
                                const __nv_bfloat16* __restrict__ direct_state,
                                const std::int32_t* __restrict__ valid_columns,
                                __nv_bfloat16* __restrict__ residual, std::int32_t c_count,
                                std::int32_t w, std::int32_t b_count) {
    const std::int64_t n = static_cast<std::int64_t>(c_count) * w * b_count;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t c = static_cast<std::int32_t>(i % c_count);
        const std::int64_t jb = i / c_count;
        const std::int32_t j = static_cast<std::int32_t>(jb % w);
        const std::int32_t b = static_cast<std::int32_t>(jb / w);
        const std::int32_t valid = valid_columns != nullptr ? valid_columns[b] : w;
        if (j >= valid) { continue; }
        const __nv_bfloat16* window =
            initial_slots != nullptr
                ? states + static_cast<std::int64_t>(initial_slots[b]) * kPleConvState * c_count
                : direct_state;
        const std::int64_t row_base = static_cast<std::int64_t>(b) * w * c_count;
        float conv                  = 0.0f;
#pragma unroll
        for (int k = 0; k < kPleConvTaps; ++k) {
            const std::int32_t at = j - (kPleConvTaps - 1 - k) * kPleConvDilation;
            const float u         = at >= 0
                                        ? __bfloat162float(normalized[row_base + static_cast<std::int64_t>(at) * c_count + c])
                                        : __bfloat162float(window[static_cast<std::int64_t>(kPleConvState + at) * c_count + c]);
            conv += __bfloat162float(weight[static_cast<std::int64_t>(k) * c_count + c]) * u;
        }
        const float silu = conv / (1.0f + __expf(-conv));
        const float out  = __bfloat162float(residual[i]) + __bfloat162float(gated[i]) + silu;
        residual[i]      = __float2bfloat16_rn(out);
    }
}

// One thread per (channel,row): load the initial window, then emit the window after each valid
// column. Reading the initial window first makes an initial slot inside the reservation safe.
__global__ void ple_state_kernel(const __nv_bfloat16* __restrict__ normalized,
                                 __nv_bfloat16* states,
                                 const std::int32_t* __restrict__ initial_slots,
                                 const std::int32_t* __restrict__ base_slots,
                                 const __nv_bfloat16* direct_in, __nv_bfloat16* direct_out,
                                 const std::int32_t* __restrict__ valid_columns,
                                 std::int32_t c_count, std::int32_t w, std::int32_t b_count,
                                 bool final_only) {
    const std::int64_t n = static_cast<std::int64_t>(c_count) * b_count;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t c     = static_cast<std::int32_t>(i % c_count);
        const std::int32_t b     = static_cast<std::int32_t>(i / c_count);
        const std::int32_t valid = valid_columns != nullptr ? valid_columns[b] : w;
        const std::int64_t slot  = static_cast<std::int64_t>(kPleConvState) * c_count;
        const __nv_bfloat16* in =
            initial_slots != nullptr ? states + initial_slots[b] * slot : direct_in;
        __nv_bfloat16 window[kPleConvState];
#pragma unroll
        for (int k = 0; k < kPleConvState; ++k) {
            window[k] = in[static_cast<std::int64_t>(k) * c_count + c];
        }
        const std::int64_t row_base = static_cast<std::int64_t>(b) * w * c_count;
        for (std::int32_t j = 0; j < valid; ++j) {
#pragma unroll
            for (int k = 0; k < kPleConvState - 1; ++k) { window[k] = window[k + 1]; }
            window[kPleConvState - 1] = normalized[row_base + static_cast<std::int64_t>(j) * c_count + c];
            if (final_only && j + 1 < valid) { continue; }
            __nv_bfloat16* out =
                base_slots != nullptr
                    ? states + (static_cast<std::int64_t>(base_slots[b]) + j) * slot
                    : direct_out;
#pragma unroll
            for (int k = 0; k < kPleConvState; ++k) {
                out[static_cast<std::int64_t>(k) * c_count + c] = window[k];
            }
        }
    }
}


// Commit fold: one thread per (channel,row).
constexpr int kFoldRows = 8;
struct FoldRows {
    std::int32_t source[kFoldRows], destination[kFoldRows], commit[kFoldRows];
};

__global__ void ple_fold_kernel(__nv_bfloat16* states, const __nv_bfloat16* __restrict__ record,
                                FoldRows rows, std::int32_t c_count, std::int32_t w,
                                std::int32_t row_count) {
    const std::int64_t n = static_cast<std::int64_t>(c_count) * row_count;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t c    = static_cast<std::int32_t>(i % c_count);
        const std::int32_t r    = static_cast<std::int32_t>(i / c_count);
        const std::int64_t slot = static_cast<std::int64_t>(kPleConvState) * c_count;
        const __nv_bfloat16* in = states + rows.source[r] * slot;
        __nv_bfloat16 window[kPleConvState];
#pragma unroll
        for (int k = 0; k < kPleConvState; ++k) { window[k] = in[static_cast<std::int64_t>(k) * c_count + c]; }
        const std::int64_t row_base = static_cast<std::int64_t>(r) * w * c_count;
        for (std::int32_t j = 0; j < rows.commit[r]; ++j) {
#pragma unroll
            for (int k = 0; k < kPleConvState - 1; ++k) { window[k] = window[k + 1]; }
            window[kPleConvState - 1] = record[row_base + static_cast<std::int64_t>(j) * c_count + c];
        }
        __nv_bfloat16* out = states + rows.destination[r] * slot;
#pragma unroll
        for (int k = 0; k < kPleConvState; ++k) { out[static_cast<std::int64_t>(k) * c_count + c] = window[k]; }
    }
}

// Width-4 causal depthwise conv + SiLU over [C,W,B] reading each row's [C,3] initial window.
__global__ void conv_record_kernel(const __nv_bfloat16* __restrict__ x,
                                   const __nv_bfloat16* __restrict__ weight,
                                   const __nv_bfloat16* __restrict__ states,
                                   const std::int32_t* __restrict__ initial_slots,
                                   const std::int32_t* __restrict__ valid_columns,
                                   __nv_bfloat16* __restrict__ out, std::int32_t c_count,
                                   std::int32_t w, std::int32_t b_count) {
    const std::int64_t n = static_cast<std::int64_t>(c_count) * w * b_count;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int32_t c  = static_cast<std::int32_t>(i % c_count);
        const std::int64_t jb = i / c_count;
        const std::int32_t j  = static_cast<std::int32_t>(jb % w);
        const std::int32_t b  = static_cast<std::int32_t>(jb / w);
        const std::int32_t valid = valid_columns != nullptr ? valid_columns[b] : w;
        if (j >= valid) {
            out[i] = __float2bfloat16_rn(0.0f);
            continue;
        }
        const __nv_bfloat16* window = states + static_cast<std::int64_t>(initial_slots[b]) * 3 * c_count;
        const std::int64_t row_base = static_cast<std::int64_t>(b) * w * c_count;
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const std::int32_t at = j - 3 + k;
            const float u = at >= 0 ? __bfloat162float(x[row_base + static_cast<std::int64_t>(at) * c_count + c])
                                    : __bfloat162float(window[static_cast<std::int64_t>(3 + at) * c_count + c]);
            acc += __bfloat162float(weight[static_cast<std::int64_t>(k) * c_count + c]) * u;
        }
        out[i] = __float2bfloat16_rn(acc / (1.0f + __expf(-acc)));
    }
}

} // namespace

void ple_dequantize_launch(const Tensor& codes, const Tensor& scale, Tensor& out,
                           cudaStream_t stream) {
    const std::int64_t n = out.numel();
    ple_dequantize_kernel<<<elementwise_grid(n), kBlock, 0, stream>>>(
        static_cast<const __nv_fp8_e4m3*>(codes.data),
        static_cast<const __nv_bfloat16*>(scale.data), static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void ple_gate_launch(const Tensor& key, const Tensor& query, const Tensor& value,
                     std::int32_t streams, Tensor& out, cudaStream_t stream) {
    const std::int64_t blocks = static_cast<std::int64_t>(streams) * value.ne[1];
    ple_gate_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key.data), static_cast<const __nv_bfloat16*>(query.data),
        static_cast<const __nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(out.data),
        value.ne[0], streams);
    CUDA_CHECK(cudaGetLastError());
}

void ple_conv_residual_launch(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                              const void* state_in, void* state_out, Tensor& residual,
                              cudaStream_t stream) {
    const std::int32_t c = residual.ne[0];
    const std::int32_t t = residual.ne[1];
    const auto* in       = static_cast<const __nv_bfloat16*>(state_in);
    const std::int64_t n = static_cast<std::int64_t>(c) * t;
    ple_conv_kernel<<<elementwise_grid(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(weight.data), nullptr, nullptr, in, nullptr,
        static_cast<__nv_bfloat16*>(residual.data), c, t, 1);
    CUDA_CHECK(cudaGetLastError());
    ple_state_kernel<<<elementwise_grid(c), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data), nullptr, nullptr, nullptr, in,
        static_cast<__nv_bfloat16*>(state_out), nullptr, c, t, 1, /*final_only=*/true);
    CUDA_CHECK(cudaGetLastError());
}

void ple_conv_residual_snapshot_launch(const Tensor& gated, const Tensor& normalized,
                                       const Tensor& weight, Tensor& states,
                                       const Tensor& valid_columns,
                                       const Tensor& initial_state_slots,
                                       const Tensor& snapshot_base_slots, Tensor& residual,
                                       cudaStream_t stream) {
    const std::int32_t c  = residual.ne[0];
    const std::int32_t w  = residual.ne[1];
    const std::int32_t b  = residual.ne[2];
    const auto* valid     = static_cast<const std::int32_t*>(valid_columns.data);
    const auto* initial   = static_cast<const std::int32_t*>(initial_state_slots.data);
    const auto* bases     = static_cast<const std::int32_t*>(snapshot_base_slots.data);
    auto* state_data      = static_cast<__nv_bfloat16*>(states.data);
    const std::int64_t n  = static_cast<std::int64_t>(c) * w * b;
    ple_conv_kernel<<<elementwise_grid(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(weight.data), state_data, initial, nullptr, valid,
        static_cast<__nv_bfloat16*>(residual.data), c, w, b);
    CUDA_CHECK(cudaGetLastError());
    ple_state_kernel<<<elementwise_grid(static_cast<std::int64_t>(c) * b), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data), state_data, initial, bases, nullptr,
        nullptr, valid, c, w, b, /*final_only=*/false);
    CUDA_CHECK(cudaGetLastError());
}

void ple_conv_residual_record_launch(const Tensor& gated, const Tensor& normalized,
                                     const Tensor& weight, const Tensor& states,
                                     const Tensor& valid_columns,
                                     const Tensor& initial_state_slots, Tensor& residual,
                                     cudaStream_t stream) {
    const std::int32_t c = residual.ne[0];
    const std::int32_t w = residual.ne[1];
    const std::int32_t b = residual.ne[2];
    const std::int64_t n = static_cast<std::int64_t>(c) * w * b;
    ple_conv_kernel<<<elementwise_grid(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(states.data),
        static_cast<const std::int32_t*>(initial_state_slots.data), nullptr,
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(residual.data), c, w, b);
    CUDA_CHECK(cudaGetLastError());
}

void ple_conv_replay_fold_launch(Tensor& states, const Tensor& record,
                                 std::span<const GdnReplayFoldRow> rows, cudaStream_t stream) {
    FoldRows packed{};
    for (std::size_t r = 0; r < rows.size(); ++r) {
        packed.source[r]      = rows[r].source_state_slot;
        packed.destination[r] = rows[r].destination_state_slot;
        packed.commit[r]      = rows[r].commit_columns;
    }
    const std::int32_t c = states.ne[0];
    const auto count     = static_cast<std::int32_t>(rows.size());
    ple_fold_kernel<<<elementwise_grid(static_cast<std::int64_t>(c) * count), kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(states.data), static_cast<const __nv_bfloat16*>(record.data),
        packed, c, record.ne[1], count);
    CUDA_CHECK(cudaGetLastError());
}

void causal_conv1d_silu_record_launch(const Tensor& x, const Tensor& weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& out,
                                      cudaStream_t stream) {
    const std::int32_t c = x.ne[0];
    const std::int32_t w = x.ne[1];
    const std::int32_t b = x.ne[2];
    const std::int64_t n = static_cast<std::int64_t>(c) * w * b;
    conv_record_kernel<<<elementwise_grid(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(out.data), c, w, b);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
