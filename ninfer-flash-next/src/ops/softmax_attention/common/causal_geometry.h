#pragma once
#include "ops/softmax_attention/common/head_mapping.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
inline constexpr int kCausalHeadDim = 256;

template <int HeadDim, int QueryHeads, int KVHeads>
struct CausalGeometry : AttentionHeadMapping<QueryHeads, KVHeads> {
    static_assert(HeadDim > 0 && HeadDim % 64 == 0);
    static constexpr int kHeadDim = HeadDim;
};

using CausalD256H24Kv4 = CausalGeometry<256, 24, 4>;
using CausalD256H16Kv2 = CausalGeometry<256, 16, 2>;
using CausalD256H24Kv2 = CausalGeometry<256, 24, 2>;

inline constexpr bool causal_geometry_registered(int query_heads, int kv_heads) {
    return (query_heads == 24 && kv_heads == 4) || (query_heads == 16 && kv_heads == 2) ||
           (query_heads == 24 && kv_heads == 2);
}

// Grouped quantized kernels pack Tokens*GroupSize query rows into at most four 16-row MMA tiles.
// The grouped/parallel token tile is the largest such count, capped at eight tokens.
inline constexpr int causal_grouped_token_tile(int group_size) {
    return 64 / group_size < 8 ? 64 / group_size : 8;
}

// Invoke f.template operator()<G>() for the registered causal geometry.
template <class F>
decltype(auto) dispatch_causal_geometry(int query_heads, int kv_heads, F&& f) {
    if (query_heads == 24 && kv_heads == 4) return f.template operator()<CausalD256H24Kv4>();
    if (query_heads == 16 && kv_heads == 2) return f.template operator()<CausalD256H16Kv2>();
    if (query_heads == 24 && kv_heads == 2) return f.template operator()<CausalD256H24Kv2>();
    throw std::invalid_argument("causal attention: unregistered head geometry");
}
} // namespace ninfer::ops::detail
