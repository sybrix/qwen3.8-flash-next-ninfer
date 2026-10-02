// Implements: include/ninfer/ops/hyper_connection.h
// Bandwidth-bound elementwise and per-group reductions over contiguous BF16 stream state.
#include "ops/launcher/hyper_connection.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock   = 256;
constexpr int kMaxGrid = 8192;

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

int elementwise_grid(std::int64_t n) {
    return static_cast<int>(std::clamp<std::int64_t>((n + kBlock - 1) / kBlock, 1, kMaxGrid));
}

// One CTA per (group, column) row of D contiguous values.
__global__ void grouped_rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ weight,
                                       __nv_bfloat16* __restrict__ out, std::int32_t d,
                                       std::int32_t groups, float eps) {
    const std::int64_t row = blockIdx.x;
    const std::int32_t g   = static_cast<std::int32_t>(row % groups);
    const auto* in         = x + row * d;
    auto* dst              = out + row * d;
    const auto* w          = weight + static_cast<std::int64_t>(g) * d;

    float sum = 0.0f;
    for (std::int32_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = __bfloat162float(in[i]);
        sum += v * v;
    }
    __shared__ float partial[kBlock / 32];
    sum = warp_reduce_sum(sum);
    if ((threadIdx.x & 31) == 0) { partial[threadIdx.x >> 5] = sum; }
    __syncthreads();
    if (threadIdx.x < 32) {
        float total = threadIdx.x < kBlock / 32 ? partial[threadIdx.x] : 0.0f;
        total       = warp_reduce_sum(total);
        if (threadIdx.x == 0) { partial[0] = total; }
    }
    __syncthreads();
    const float inv = rsqrtf(partial[0] / static_cast<float>(d) + eps);
    for (std::int32_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = __bfloat162float(in[i]) * inv * (1.0f + __bfloat162float(w[i]));
        dst[i]        = __float2bfloat16_rn(v);
    }
}

__global__ void hyper_connection_mix_kernel(const __nv_bfloat16* __restrict__ normalized,
                                            const __nv_bfloat16* __restrict__ logits,
                                            __nv_bfloat16* __restrict__ out, std::int32_t h,
                                            std::int32_t streams, std::int64_t n) {
    const float scale = 1.0f / static_cast<float>(streams);
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t t   = i / h;
        const std::int32_t col = static_cast<std::int32_t>(i - t * h);
        const std::int64_t base = t * static_cast<std::int64_t>(streams) * h + col;
        float acc              = 0.0f;
        for (std::int32_t s = 0; s < streams; ++s) {
            const std::int64_t at = base + static_cast<std::int64_t>(s) * h;
            acc += sigmoidf_(__bfloat162float(logits[at])) * __bfloat162float(normalized[at]);
        }
        out[i] = __float2bfloat16_rn(acc * scale);
    }
}

__global__ void hyper_connection_hidden_kernel(const __nv_bfloat16* __restrict__ down,
                                               __nv_bfloat16* __restrict__ hidden, float inv_streams,
                                               std::int64_t n) {
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const float v = __bfloat162float(down[i]) * inv_streams;
        hidden[i]     = __float2bfloat16_rn(v * sigmoidf_(v));
    }
}

__global__ void hyper_connection_inject_weights_kernel(const __nv_bfloat16* __restrict__ logits,
                                                       float* __restrict__ inject, float inv_streams,
                                                       std::int64_t n) {
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        inject[i] = 2.0f * sigmoidf_(__bfloat162float(logits[i]) * inv_streams);
    }
}

__global__ void hyper_connection_inject_kernel(const __nv_bfloat16* __restrict__ y,
                                               const float* __restrict__ inject,
                                               __nv_bfloat16* __restrict__ state, std::int32_t h,
                                               std::int32_t streams, std::int64_t n) {
    const std::int64_t width = static_cast<std::int64_t>(streams) * h;
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t t   = i / width;
        const std::int64_t r   = i - t * width;
        const std::int32_t s   = static_cast<std::int32_t>(r / h);
        const std::int64_t col = r - static_cast<std::int64_t>(s) * h;
        const float v          = __bfloat162float(state[i]) +
                        inject[t * streams + s] * __bfloat162float(y[t * h + col]);
        state[i] = __float2bfloat16_rn(v);
    }
}

} // namespace

void grouped_rmsnorm_launch(const Tensor& x, const Tensor& weight, std::int32_t groups, float eps,
                            Tensor& out, cudaStream_t stream) {
    const std::int32_t d    = x.ne[0] / groups;
    const std::int64_t rows = x.numel() / d;
    grouped_rmsnorm_kernel<<<static_cast<unsigned>(rows), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<__nv_bfloat16*>(out.data), d, groups, eps);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_mix_launch(const Tensor& normalized, const Tensor& logits,
                                 std::int32_t streams, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    hyper_connection_mix_kernel<<<elementwise_grid(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<__nv_bfloat16*>(out.data),
        out.ne[0], streams, n);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_gates_launch(const Tensor& down, const Tensor& inject_logits,
                                   std::int32_t streams, Tensor& hidden, Tensor& inject,
                                   cudaStream_t stream) {
    const float inv_streams = 1.0f / static_cast<float>(streams);
    if (down.data != nullptr) {
        const std::int64_t n = down.numel();
        hyper_connection_hidden_kernel<<<elementwise_grid(n), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(down.data), static_cast<__nv_bfloat16*>(hidden.data),
            inv_streams, n);
        CUDA_CHECK(cudaGetLastError());
    }
    if (inject_logits.data != nullptr) {
        const std::int64_t n = inject_logits.numel();
        hyper_connection_inject_weights_kernel<<<elementwise_grid(n), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inject_logits.data), static_cast<float*>(inject.data),
            inv_streams, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

void hyper_connection_inject_launch(const Tensor& y, const Tensor& inject, Tensor& state,
                                    cudaStream_t stream) {
    const std::int64_t n        = state.numel();
    const std::int32_t streams  = inject.ne[0];
    hyper_connection_inject_kernel<<<elementwise_grid(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(y.data), static_cast<const float*>(inject.data),
        static_cast<__nv_bfloat16*>(state.data), y.ne[0], streams, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
