#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class Nvfp4KvFamily { Grouped, ParallelGrouped, Tiled };

struct Nvfp4KvCausalPlan {
    Nvfp4KvFamily family;
    int query_heads, kv_heads, width, batch, query_tile;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

Nvfp4KvCausalPlan make_nvfp4_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope);
std::size_t nvfp4_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                     int max_width, CausalAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
