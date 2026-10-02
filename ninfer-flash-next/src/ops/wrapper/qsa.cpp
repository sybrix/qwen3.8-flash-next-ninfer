// ninfer::ops - QSA wrappers: validate the public contracts and dispatch to the launcher.
#include "ninfer/ops/qsa.h"

#include "ops/launcher/qsa.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
             const char* op, const char* name) {
    int i = 0;
    bool ok = t.dtype == dtype && t.is_contiguous() && t.data != nullptr;
    for (const auto extent : shape) { ok = ok && t.ne[i++] == extent; }
    for (; i < 4; ++i) { ok = ok && t.ne[i] == 1; }
    if (!ok) { throw std::invalid_argument(std::string(op) + ": invalid " + name); }
}

void require_rope(std::int32_t rotary_dim, std::int32_t di, float theta, float eps,
                  const char* op) {
    if (rotary_dim <= 0 || rotary_dim % 2 || rotary_dim > di || !(theta > 0.0f) ||
        !std::isfinite(theta) || !(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument(std::string(op) + ": invalid RoPE or norm parameters");
    }
}

void require_index_cache(const Tensor& pages, const Tensor& tables, std::int32_t di,
                         const char* op) {
    if (pages.dtype != DType::BF16 || pages.ne[0] != di || pages.ne[1] != kPagedKVPageSize ||
        pages.ne[2] != 1 || !pages.is_contiguous() || pages.data == nullptr ||
        tables.dtype != DType::I32 || tables.ne[2] != 1 || !tables.is_contiguous() ||
        tables.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": invalid index cache or block tables");
    }
}

void require_position_pages(const Tensor& positions, const Tensor& index_pages, const char* op) {
    if (positions.dtype != DType::I32 || positions.ne[0] != 4 ||
        positions.ne[1] != kPagedKVPageSize || positions.ne[2] != 1 ||
        positions.ne[3] != index_pages.ne[3] || !positions.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": position pages must be I32 [4,P,1,pages]");
    }
}

} // namespace

void qsa_prepare_query(const Tensor& q, const Tensor& weight, const Tensor& positions,
                       std::int32_t rotary_dim, float theta, float eps, Tensor& out,
                       cudaStream_t stream) {
    constexpr const char* op = "qsa_prepare_query";
    const auto di = q.ne[0], heads = q.ne[1], width = q.ne[2], seqs = q.ne[3];
    if (di <= 0 || di > 128 || heads <= 0 || width <= 0 || seqs <= 0) {
        throw std::invalid_argument("qsa_prepare_query: q must be BF16 [Di<=128,H,W,B]");
    }
    require_rope(rotary_dim, di, theta, eps, op);
    require(q, DType::BF16, {di, heads, width, seqs}, op, "q");
    require(out, DType::BF16, {di, heads, width, seqs}, op, "out");
    require(weight, DType::BF16, {di}, op, "weight");
    if (positions.dtype != DType::I32 || !positions.is_contiguous() || positions.data == nullptr ||
        (positions.numel() != std::int64_t(width) * seqs &&
         positions.numel() != 3 * std::int64_t(width) * seqs)) {
        throw std::invalid_argument("qsa_prepare_query: positions must be I32 [W*B] or [W*B,3]");
    }
    if (q.data == out.data) { throw std::invalid_argument("qsa_prepare_query: out aliases q"); }
    detail::qsa_prepare_query_launch(q, weight, positions, rotary_dim, theta, eps, out, stream);
}

void qsa_index_append(const Tensor& keys, const Tensor& positions, const Tensor& rows,
                      const Tensor& index_pages, const Tensor& block_tables, cudaStream_t stream) {
    qsa_index_append(keys, positions, rows, index_pages, block_tables, Tensor{}, Tensor{}, stream);
}

void qsa_index_append(const Tensor& keys, const Tensor& positions, const Tensor& rows,
                      const Tensor& index_pages, const Tensor& block_tables,
                      const Tensor& rope_positions, const Tensor& position_pages,
                      cudaStream_t stream) {
    constexpr const char* op = "qsa_index_append";
    const auto di = keys.ne[0], width = keys.ne[1], seqs = keys.ne[2];
    if (di <= 0 || width <= 0 || seqs <= 0) {
        throw std::invalid_argument("qsa_index_append: keys must be BF16 [Di,W,B]");
    }
    require(keys, DType::BF16, {di, width, seqs}, op, "keys");
    require(positions, DType::I32, {width, seqs}, op, "positions");
    require(rows, DType::I32, {seqs}, op, "rows");
    require_index_cache(index_pages, block_tables, di, op);
    if (position_pages.data != nullptr) {
        require_position_pages(position_pages, index_pages, op);
        if (rope_positions.dtype != DType::I32 || !rope_positions.is_contiguous() ||
            rope_positions.data == nullptr ||
            (rope_positions.numel() != std::int64_t(width) * seqs &&
             rope_positions.numel() != 3 * std::int64_t(width) * seqs)) {
            throw std::invalid_argument("qsa_index_append: rope positions must be I32 [W*B] or [W*B,3]");
        }
    }
    detail::qsa_index_append_launch(keys, positions, rows, index_pages, block_tables,
                                    rope_positions, position_pages, stream);
}

std::size_t qsa_select_workspace_bytes(const QsaSelectGeometry& geometry, std::int32_t index_dim,
                                       std::int32_t sequences, std::int32_t columns,
                                       std::int32_t max_visible) {
    if (geometry.compress_ratio <= 0 || geometry.block_budget <= 0 || index_dim <= 0 ||
        sequences <= 0 || columns <= 0 || max_visible <= 0) {
        throw std::invalid_argument("qsa_select workspace: invalid geometry");
    }
    return detail::qsa_select_workspace_launch_bytes(geometry.compress_ratio, index_dim,
                                                     sequences, columns, max_visible);
}

void qsa_select(const Tensor& query, const Tensor& positions, const Tensor& rows,
                const Tensor& index_pages, const Tensor& block_tables, const Tensor& key_norm,
                const QsaSelectGeometry& geometry, std::int32_t max_visible,
                WorkspaceArena& workspace, Tensor& selected, Tensor& counts, cudaStream_t stream) {
    qsa_select(query, positions, rows, index_pages, Tensor{}, block_tables, key_norm, geometry,
               max_visible, workspace, selected, counts, stream);
}

void qsa_select(const Tensor& query, const Tensor& positions, const Tensor& rows,
                const Tensor& index_pages, const Tensor& position_pages,
                const Tensor& block_tables, const Tensor& key_norm,
                const QsaSelectGeometry& geometry, std::int32_t max_visible,
                WorkspaceArena& workspace, Tensor& selected, Tensor& counts, cudaStream_t stream) {
    constexpr const char* op = "qsa_select";
    const auto di = query.ne[0], heads = query.ne[1], width = query.ne[2], seqs = query.ne[3];
    if (di <= 0 || di > 128 || heads <= 0 || width <= 0 || seqs <= 0 || max_visible <= 0) {
        throw std::invalid_argument("qsa_select: query must be BF16 [Di<=128,H,W,B]");
    }
    require_rope(geometry.rotary_dim, di, geometry.theta, geometry.eps, op);
    require(query, DType::BF16, {di, heads, width, seqs}, op, "query");
    require(positions, DType::I32, {width, seqs}, op, "positions");
    require(rows, DType::I32, {seqs}, op, "rows");
    require(key_norm, DType::BF16, {di}, op, "key_norm");
    require(selected, DType::I32, {geometry.max_selected(), width, seqs}, op, "selected");
    require(counts, DType::I32, {width, seqs}, op, "counts");
    require_index_cache(index_pages, block_tables, di, op);
    if (position_pages.data != nullptr) { require_position_pages(position_pages, index_pages, op); }
    const std::size_t bytes =
        qsa_select_workspace_bytes(geometry, di, seqs, width * seqs, max_visible);
    if (workspace.base() == nullptr || workspace.used() > workspace.capacity() ||
        workspace.capacity() - workspace.used() < bytes) {
        throw std::invalid_argument("qsa_select: insufficient workspace capacity");
    }
    auto scope = workspace.scope();
    void* scratch = workspace.alloc_bytes(bytes).data;
    detail::qsa_select_launch(query, positions, rows, index_pages, position_pages, block_tables,
                              key_norm,
                              geometry.compress_ratio, geometry.block_budget, geometry.rotary_dim,
                              geometry.theta, geometry.eps, max_visible, scratch, selected, counts,
                              stream);
}

void qsa_sparse_attention(const Tensor& q, const Tensor& selected, const Tensor& counts,
                          const Tensor& rows, const PagedKVBatchLayerView& cache, float scale,
                          Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "qsa_sparse_attention";
    const auto d = q.ne[0], heads = q.ne[1], width = q.ne[2], seqs = q.ne[3];
    if (d != 256 || cache.head_dim != 256 || cache.num_kv_heads <= 0 ||
        heads % cache.num_kv_heads || heads / cache.num_kv_heads > 16 || width <= 0 ||
        seqs <= 0 || !(scale > 0.0f)) {
        throw std::invalid_argument("qsa_sparse_attention: unsupported head geometry");
    }
    if (cache.storage != KvCacheStorage::BFloat16 &&
        cache.storage != KvCacheStorage::Fp8E4M3Row256) {
        throw std::invalid_argument("qsa_sparse_attention: KV storage must be bf16 or fp8");
    }
    require(q, DType::BF16, {d, heads, width, seqs}, op, "q");
    require(out, DType::BF16, {d, heads, width, seqs}, op, "out");
    require(counts, DType::I32, {width, seqs}, op, "counts");
    require(rows, DType::I32, {seqs}, op, "rows");
    if (selected.dtype != DType::I32 || selected.ne[1] != width || selected.ne[2] != seqs ||
        !selected.is_contiguous()) {
        throw std::invalid_argument("qsa_sparse_attention: invalid selected");
    }
    detail::qsa_sparse_attention_launch(q, selected, counts, rows, cache, scale, out, stream);
}

} // namespace ninfer::ops
