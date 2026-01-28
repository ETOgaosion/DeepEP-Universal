#include "configs.cuh"
#include "buffer.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#ifndef DISABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#include <cuda_fp16.h>

namespace deep_ep {

namespace intranode {

#ifndef DISABLE_NVSHMEM

namespace {

__device__ __forceinline__ int* get_raw_counts_ptr(void* base_ptr, int num_ranks) {
    (void)num_ranks;
    return reinterpret_cast<int*>(base_ptr);
}

__device__ __forceinline__ int* get_rank_prefix_ptr(void* base_ptr) {
    return reinterpret_cast<int*>(base_ptr);
}

__device__ __forceinline__ int* get_raw_expert_counts_ptr(void* base_ptr, int num_ranks, int num_experts) {
    auto offset = static_cast<int64_t>(num_ranks) * num_ranks * sizeof(int);
    return reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(base_ptr) + offset);
}

struct ChannelBuffers {
    int* channel_head;
    int4* x;
    int* src_idx;
    int64_t* topk_idx;
    float* topk_weights;
    float* x_scales;
};

__device__ __forceinline__ ChannelBuffers get_channel_buffers(void* base_ptr, int num_ranks, int num_experts,
                                                              int num_channels, int num_recv_buffer_tokens,
                                                              int hidden_int4, int num_topk, int num_scales,
                                                              int channel_rank_offset) {
    int64_t offset = 0;
    offset += static_cast<int64_t>(num_ranks) * num_ranks * sizeof(int);
    offset += static_cast<int64_t>(num_experts) * sizeof(int);

    const int64_t channel_meta_elems = static_cast<int64_t>(num_channels) * num_ranks;
    const int64_t channel_meta_bytes = channel_meta_elems * sizeof(int);
    const int64_t channel_head_offset = offset + channel_meta_bytes * 2;

    const int64_t data_offset = offset + channel_meta_bytes * 4;

    const int64_t x_bytes_per_channel_rank = static_cast<int64_t>(num_recv_buffer_tokens) * hidden_int4 * sizeof(int4);
    const int64_t src_idx_bytes_per_channel_rank = static_cast<int64_t>(num_recv_buffer_tokens) * sizeof(int);
    const int64_t topk_idx_bytes_per_channel_rank = static_cast<int64_t>(num_recv_buffer_tokens) * num_topk * sizeof(int64_t);
    const int64_t topk_weights_bytes_per_channel_rank = static_cast<int64_t>(num_recv_buffer_tokens) * num_topk * sizeof(float);
    const int64_t x_scales_bytes_per_channel_rank = static_cast<int64_t>(num_recv_buffer_tokens) * num_scales * sizeof(float);

    const int64_t x_section_bytes = x_bytes_per_channel_rank * channel_meta_elems;
    const int64_t src_idx_section_bytes = src_idx_bytes_per_channel_rank * channel_meta_elems;
    const int64_t topk_idx_section_bytes = topk_idx_bytes_per_channel_rank * channel_meta_elems;
    const int64_t topk_weights_section_bytes = topk_weights_bytes_per_channel_rank * channel_meta_elems;

    auto* base_u8 = reinterpret_cast<uint8_t*>(base_ptr);
    auto* channel_head = reinterpret_cast<int*>(base_u8 + channel_head_offset) + channel_rank_offset;
    auto* x_base = base_u8 + data_offset;
    auto* src_idx_base = x_base + x_section_bytes;
    auto* topk_idx_base = src_idx_base + src_idx_section_bytes;
    auto* topk_weights_base = topk_idx_base + topk_idx_section_bytes;
    auto* x_scales_base = topk_weights_base + topk_weights_section_bytes;

    ChannelBuffers buffers;
    buffers.channel_head = channel_head;
    buffers.x = reinterpret_cast<int4*>(x_base + x_bytes_per_channel_rank * channel_rank_offset);
    buffers.src_idx = reinterpret_cast<int*>(src_idx_base + src_idx_bytes_per_channel_rank * channel_rank_offset);
    buffers.topk_idx = reinterpret_cast<int64_t*>(topk_idx_base + topk_idx_bytes_per_channel_rank * channel_rank_offset);
    buffers.topk_weights = reinterpret_cast<float*>(topk_weights_base + topk_weights_bytes_per_channel_rank * channel_rank_offset);
    buffers.x_scales = reinterpret_cast<float*>(x_scales_base + x_scales_bytes_per_channel_rank * channel_rank_offset);
    return buffers;
}

__global__ void write_local_counts(const int* num_tokens_per_rank, const int* num_tokens_per_expert,
                                   int num_ranks, int num_experts, void** buffer_ptrs) {
    auto* base_ptr = buffer_ptrs[0];
    auto* raw_counts = get_raw_counts_ptr(base_ptr, num_ranks);
    auto* raw_expert_counts = get_raw_expert_counts_ptr(base_ptr, num_ranks, num_experts);

    int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < num_ranks; i += static_cast<int>(blockDim.x))
        raw_counts[i] = num_tokens_per_rank[i];
    for (int i = tid; i < num_experts; i += static_cast<int>(blockDim.x))
        raw_expert_counts[i] = num_tokens_per_expert[i];
}

__global__ void gather_counts(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                              const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                              int expert_alignment, int* rank_prefix_matrix_copy, int rank, void** buffer_ptrs) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    (void)num_tokens_per_rank;
    (void)num_tokens_per_expert;
    auto* base_ptr = buffer_ptrs[0];
    auto* raw_counts = get_raw_counts_ptr(base_ptr, num_ranks);
    auto* rank_prefix_matrix_local = get_rank_prefix_ptr(base_ptr);
    auto* raw_expert_counts = get_raw_expert_counts_ptr(base_ptr, num_ranks, num_experts);

    for (int dest = 0; dest < num_ranks; ++ dest) {
        int prefix = 0;
        for (int src = 0; src < num_ranks; ++ src) {
            int value = nvshmem_int_g(raw_counts + dest, src);
            prefix += value;
            int idx = src * num_ranks + dest;
            rank_prefix_matrix_copy[idx] = prefix;
            rank_prefix_matrix_local[idx] = prefix;
        }
        if (dest == rank)
            *moe_recv_counter_mapped = prefix;
    }

    int num_experts_per_rank = num_experts / num_ranks;
    for (int i = 0; i < num_experts_per_rank; ++ i) {
        int expert_idx = rank * num_experts_per_rank + i;
        int sum = 0;
        for (int src = 0; src < num_ranks; ++ src)
            sum += nvshmem_int_g(raw_expert_counts + expert_idx, src);
        sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
        moe_recv_expert_counter_mapped[i] = sum;
    }
}

__global__ void clear_channel_head(int num_ranks, int num_experts, int num_channels,
                                   int num_recv_buffer_tokens, int hidden_int4,
                                   int num_topk, int num_scales, void** buffer_ptrs) {
    int tid = static_cast<int>(threadIdx.x);
    if (tid >= num_ranks)
        return;
    auto* base_ptr = buffer_ptrs[0];
    int channel_rank_offset = tid; // channel 0
    auto buffers = get_channel_buffers(base_ptr, num_ranks, num_experts, num_channels,
                                       num_recv_buffer_tokens, hidden_int4, num_topk, num_scales,
                                       channel_rank_offset);
    buffers.channel_head[0] = 0;
}

__global__ void dispatch_send(void** buffer_ptrs, int rank, int num_ranks, int num_experts, int num_channels,
                              const bool* is_token_in_rank,
                              const int4* x, const float* x_scales,
                              const int64_t* topk_idx, const float* topk_weights,
                              int num_tokens, int num_topk, int num_scales,
                              int hidden_int4, int scale_token_stride, int scale_hidden_stride,
                              int num_recv_buffer_tokens) {
    int token_idx = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (token_idx >= num_tokens)
        return;

    auto* base_ptr = buffer_ptrs[0];
    auto* rank_prefix_matrix = get_rank_prefix_ptr(base_ptr);
    int num_experts_per_rank = num_experts / num_ranks;
    for (int dest = 0; dest < num_ranks; ++ dest) {
        if (!is_token_in_rank[token_idx * num_ranks + dest])
            continue;

        int channel_rank_offset = dest; // channel 0
        auto buffers = get_channel_buffers(base_ptr, num_ranks, num_experts, num_channels,
                                           num_recv_buffer_tokens, hidden_int4, num_topk, num_scales,
                                           channel_rank_offset);

        int base_offset = (rank == 0) ? 0 : rank_prefix_matrix[(rank - 1) * num_ranks + dest];
        int local_offset = atomicAdd(buffers.channel_head, 1);
        int slot = base_offset + local_offset;
        if (slot >= num_recv_buffer_tokens)
            return;

        auto* dst_x = buffers.x + static_cast<int64_t>(slot) * hidden_int4;
        auto* src_x = x + static_cast<int64_t>(token_idx) * hidden_int4;
        nvshmem_putmem(dst_x, src_x, static_cast<size_t>(hidden_int4) * sizeof(int4), dest);

        if (buffers.src_idx != nullptr)
            nvshmem_putmem(buffers.src_idx + slot, &token_idx, sizeof(int), dest);

        if (num_topk > 0 && topk_idx != nullptr && topk_weights != nullptr) {
            int recv_expert_begin = dest * num_experts_per_rank;
            int recv_expert_end = recv_expert_begin + num_experts_per_rank;
            for (int k = 0; k < num_topk; ++ k) {
                int64_t idx_value = topk_idx[token_idx * num_topk + k];
                if (idx_value >= recv_expert_begin && idx_value < recv_expert_end) {
                    idx_value = idx_value - recv_expert_begin;
                } else {
                    idx_value = -1;
                }
                float weight_value = (idx_value >= 0) ? topk_weights[token_idx * num_topk + k] : 0.0f;
                nvshmem_putmem(buffers.topk_idx + static_cast<int64_t>(slot) * num_topk + k,
                               &idx_value, sizeof(int64_t), dest);
                nvshmem_putmem(buffers.topk_weights + static_cast<int64_t>(slot) * num_topk + k,
                               &weight_value, sizeof(float), dest);
            }
        }

        if (num_scales > 0 && x_scales != nullptr) {
            float tmp_scales[128];
            for (int i = 0; i < num_scales; ++ i) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                tmp_scales[i] = x_scales[offset];
            }
            nvshmem_putmem(buffers.x_scales + static_cast<int64_t>(slot) * num_scales, tmp_scales,
                           static_cast<size_t>(num_scales) * sizeof(float), dest);
        }
    }
}

__global__ void dispatch_copy_local(void** buffer_ptrs, int rank, int num_ranks, int num_experts, int num_channels,
                                    int4* recv_x, float* recv_x_scales,
                                    int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights,
                                    int num_topk, int num_scales, int hidden_int4, int num_recv_buffer_tokens) {
    auto* base_ptr = buffer_ptrs[0];
    auto* rank_prefix_matrix = get_rank_prefix_ptr(base_ptr);
    int total_recv = rank_prefix_matrix[(num_ranks - 1) * num_ranks + rank];
    int slot = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (slot >= total_recv)
        return;

    int channel_rank_offset = rank; // channel 0, local rank
    auto buffers = get_channel_buffers(base_ptr, num_ranks, num_experts, num_channels,
                                       num_recv_buffer_tokens, hidden_int4, num_topk, num_scales,
                                       channel_rank_offset);
    auto* src_x = buffers.x + static_cast<int64_t>(slot) * hidden_int4;
    auto* dst_x = recv_x + static_cast<int64_t>(slot) * hidden_int4;
    for (int i = 0; i < hidden_int4; ++ i)
        dst_x[i] = src_x[i];

    if (recv_src_idx != nullptr)
        recv_src_idx[slot] = buffers.src_idx[slot];

    if (num_topk > 0 && recv_topk_idx != nullptr && recv_topk_weights != nullptr) {
        auto* src_idx = buffers.topk_idx + static_cast<int64_t>(slot) * num_topk;
        auto* src_weights = buffers.topk_weights + static_cast<int64_t>(slot) * num_topk;
        auto* dst_idx = recv_topk_idx + static_cast<int64_t>(slot) * num_topk;
        auto* dst_weights = recv_topk_weights + static_cast<int64_t>(slot) * num_topk;
        for (int i = 0; i < num_topk; ++ i) {
            dst_idx[i] = src_idx[i];
            dst_weights[i] = src_weights[i];
        }
    }

    if (num_scales > 0 && recv_x_scales != nullptr) {
        auto* src_scales = buffers.x_scales + static_cast<int64_t>(slot) * num_scales;
        auto* dst_scales = recv_x_scales + static_cast<int64_t>(slot) * num_scales;
        for (int i = 0; i < num_scales; ++ i)
            dst_scales[i] = src_scales[i];
    }
}

template <typename dtype_t>
__global__ void init_output(dtype_t* out, const dtype_t* bias_0, const dtype_t* bias_1,
                            int num_tokens, int hidden) {
    int idx = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    int total = num_tokens * hidden;
    if (idx >= total)
        return;
    dtype_t value = static_cast<dtype_t>(0);
    if (bias_0 != nullptr)
        value = bias_0[idx];
    if (bias_1 != nullptr)
        value = static_cast<dtype_t>(value + bias_1[idx]);
    out[idx] = value;
}

template <typename dtype_t>
__device__ __forceinline__ void atomic_add(dtype_t* addr, dtype_t value);

template <>
__device__ __forceinline__ void atomic_add<float>(float* addr, float value) {
    atomicAdd(addr, value);
}

template <>
__device__ __forceinline__ void atomic_add<half>(half* addr, half value) {
    atomicAdd(addr, value);
}

template <>
__device__ __forceinline__ void atomic_add<nv_bfloat16>(nv_bfloat16* addr, nv_bfloat16 value) {
    atomicAdd(addr, value);
}

__global__ void clear_src_idx_buffer(void** buffer_ptrs, int rank, int num_ranks, int num_experts, int num_channels,
                                     int num_recv_buffer_tokens, int hidden_int4, int num_topk, int num_scales) {
    int slot = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (slot >= num_recv_buffer_tokens)
        return;
    auto* base_ptr = buffer_ptrs[0];
    int channel_rank_offset = rank;
    auto buffers = get_channel_buffers(base_ptr, num_ranks, num_experts, num_channels,
                                       num_recv_buffer_tokens, hidden_int4, num_topk, num_scales,
                                       channel_rank_offset);
    buffers.src_idx[slot] = -1;
}

__global__ void combine_send(void** buffer_ptrs, int rank, int num_ranks, int num_experts, int num_channels,
                             const int4* x, const float* topk_weights,
                             const int* src_idx, int num_tokens, int num_topk, int hidden_int4,
                             int num_recv_buffer_tokens) {
    int token_idx = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (token_idx >= num_tokens)
        return;

    auto* base_ptr = buffer_ptrs[0];
    auto* rank_prefix_matrix = get_rank_prefix_ptr(base_ptr);
    int src_rank = 0;
    while (src_rank < num_ranks && token_idx >= rank_prefix_matrix[src_rank * num_ranks + rank])
        ++ src_rank;
    if (src_rank >= num_ranks)
        return;

    int channel_rank_offset = src_rank;
    auto buffers = get_channel_buffers(base_ptr, num_ranks, num_experts, num_channels,
                                       num_recv_buffer_tokens, hidden_int4, num_topk, 0,
                                       channel_rank_offset);

    int base_offset = 0;
    for (int s = 0; s < rank; ++ s) {
        int count = rank_prefix_matrix[src_rank * num_ranks + s];
        if (src_rank > 0)
            count -= rank_prefix_matrix[(src_rank - 1) * num_ranks + s];
        base_offset += count;
    }

    int local_offset = atomicAdd(buffers.channel_head, 1);
    int slot = base_offset + local_offset;
    if (slot >= num_recv_buffer_tokens)
        return;

    auto* dst_x = buffers.x + static_cast<int64_t>(slot) * hidden_int4;
    auto* src_x = x + static_cast<int64_t>(token_idx) * hidden_int4;
    nvshmem_putmem(dst_x, src_x, static_cast<size_t>(hidden_int4) * sizeof(int4), src_rank);

    int idx_value = src_idx[token_idx];
    nvshmem_putmem(buffers.src_idx + slot, &idx_value, sizeof(int), src_rank);

    if (num_topk > 0 && topk_weights != nullptr) {
        nvshmem_putmem(buffers.topk_weights + static_cast<int64_t>(slot) * num_topk,
                       topk_weights + static_cast<int64_t>(token_idx) * num_topk,
                       static_cast<size_t>(num_topk) * sizeof(float), src_rank);
    }
}

template <typename dtype_t>
__global__ void combine_reduce(void** buffer_ptrs, int rank, int num_ranks, int num_experts, int num_channels,
                               dtype_t* out, float* out_topk_weights, int num_recv_tokens, int hidden,
                               int num_topk, int hidden_int4, int num_recv_buffer_tokens) {
    int slot = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (slot >= num_recv_buffer_tokens)
        return;

    auto* base_ptr = buffer_ptrs[0];
    int channel_rank_offset = rank;
    auto buffers = get_channel_buffers(base_ptr, num_ranks, num_experts, num_channels,
                                       num_recv_buffer_tokens, hidden_int4, num_topk, 0,
                                       channel_rank_offset);

    int dst_idx = buffers.src_idx[slot];
    if (dst_idx < 0 || dst_idx >= num_recv_tokens)
        return;

    auto* src_x = reinterpret_cast<const dtype_t*>(buffers.x + static_cast<int64_t>(slot) * hidden_int4);
    auto* dst_x = out + static_cast<int64_t>(dst_idx) * hidden;
    for (int i = 0; i < hidden; ++ i)
        atomic_add(dst_x + i, src_x[i]);

    if (num_topk > 0 && out_topk_weights != nullptr) {
        auto* src_w = buffers.topk_weights + static_cast<int64_t>(slot) * num_topk;
        auto* dst_w = out_topk_weights + static_cast<int64_t>(dst_idx) * num_topk;
        for (int k = 0; k < num_topk; ++ k)
            atomicAdd(dst_w + k, src_w[k]);
    }
}

__global__ void fill_int(int* ptr, int value, int count) {
    int idx = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (idx < count)
        ptr[idx] = value;
}

__global__ void fill_float(float* ptr, float value, int count) {
    int idx = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (idx < count)
        ptr[idx] = value;
}

__global__ void copy_rank_prefix_to_buffer(const int* rank_prefix_matrix, int num_ranks, void** buffer_ptrs) {
    int idx = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    int total = num_ranks * num_ranks;
    if (idx >= total)
        return;
    auto* base_ptr = buffer_ptrs[0];
    auto* rank_prefix_matrix_local = get_rank_prefix_ptr(base_ptr);
    rank_prefix_matrix_local[idx] = rank_prefix_matrix[idx];
}

} // namespace

#endif // DISABLE_NVSHMEM

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     int num_tokens, const bool* is_token_in_rank, int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy, int num_memset_int, int expert_alignment,
                     void** buffer_ptrs, int** barrier_signal_ptrs, int rank,
                     cudaStream_t stream, int num_channels) {
#ifndef DISABLE_NVSHMEM
    (void)num_tokens;
    (void)is_token_in_rank;
    (void)num_memset_int;
    (void)barrier_signal_ptrs;
    (void)num_channels;

    SETUP_LAUNCH_CONFIG(1, 256, stream);
    LAUNCH_KERNEL(&cfg, write_local_counts,
                  num_tokens_per_rank, num_tokens_per_expert, num_ranks, num_experts, buffer_ptrs);

    nvshmemx_barrier_all_on_stream(stream);

    SETUP_LAUNCH_CONFIG(1, 1, stream);
    LAUNCH_KERNEL(&cfg, gather_counts,
                  num_tokens_per_rank, moe_recv_counter_mapped, num_ranks,
                  num_tokens_per_expert, moe_recv_expert_counter_mapped, num_experts,
                  expert_alignment, rank_prefix_matrix_copy, rank, buffer_ptrs);

    SETUP_LAUNCH_CONFIG(ceil_div(num_ranks * num_channels, 256), 256, stream);
    LAUNCH_KERNEL(&cfg, fill_int, channel_prefix_matrix, 0, num_ranks * num_channels);
#else
    (void)num_tokens_per_rank;
    (void)moe_recv_counter_mapped;
    (void)num_ranks;
    (void)num_tokens_per_expert;
    (void)moe_recv_expert_counter_mapped;
    (void)num_experts;
    (void)num_tokens;
    (void)is_token_in_rank;
    (void)channel_prefix_matrix;
    (void)rank_prefix_matrix_copy;
    (void)num_memset_int;
    (void)expert_alignment;
    (void)buffer_ptrs;
    (void)barrier_signal_ptrs;
    (void)rank;
    (void)stream;
    (void)num_channels;
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

void cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int,
                            void** buffer_ptrs, int** barrier_signal_ptrs,
                            int rank, int num_ranks, cudaStream_t stream) {
#ifndef DISABLE_NVSHMEM
    (void)num_memset_int;
    (void)barrier_signal_ptrs;
    (void)rank;
    SETUP_LAUNCH_CONFIG(ceil_div(num_ranks * num_ranks, 256), 256, stream);
    LAUNCH_KERNEL(&cfg, copy_rank_prefix_to_buffer, rank_prefix_matrix, num_ranks, buffer_ptrs);
#else
    (void)rank_prefix_matrix;
    (void)num_memset_int;
    (void)buffer_ptrs;
    (void)barrier_signal_ptrs;
    (void)rank;
    (void)num_ranks;
    (void)stream;
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

void dispatch(void* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights, int* recv_channel_offset,
              int* send_head, const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
              const bool* is_token_in_rank, const int* channel_prefix_matrix,
              int num_tokens, int num_worst_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
              int scale_token_stride, int scale_hidden_stride,
              void** buffer_ptrs, int rank, int num_ranks,
              cudaStream_t stream, int num_sms,
              int num_max_send_tokens, int num_recv_buffer_tokens) {
#ifndef DISABLE_NVSHMEM
    (void)num_sms;
    (void)num_max_send_tokens;
    (void)channel_prefix_matrix;
    (void)num_worst_tokens;
    const int num_channels = num_sms > 0 ? num_sms / 2 : 1;

    SETUP_LAUNCH_CONFIG(1, 256, stream);
    LAUNCH_KERNEL(&cfg, clear_channel_head,
                  num_ranks, num_experts, num_channels, num_recv_buffer_tokens, hidden_int4, num_topk, num_scales,
                  buffer_ptrs);

    SETUP_LAUNCH_CONFIG(ceil_div(num_tokens, 256), 256, stream);
    LAUNCH_KERNEL(&cfg, dispatch_send,
                  buffer_ptrs, rank, num_ranks, num_experts, num_channels,
                  is_token_in_rank,
                  reinterpret_cast<const int4*>(x), x_scales, topk_idx, topk_weights,
                  num_tokens, num_topk, num_scales, hidden_int4, scale_token_stride, scale_hidden_stride,
                  num_recv_buffer_tokens);

    nvshmemx_barrier_all_on_stream(stream);

    SETUP_LAUNCH_CONFIG(ceil_div(num_recv_buffer_tokens, 256), 256, stream);
    LAUNCH_KERNEL(&cfg, dispatch_copy_local,
                  buffer_ptrs, rank, num_ranks, num_experts, num_channels,
                  reinterpret_cast<int4*>(recv_x), recv_x_scales,
                  recv_src_idx, recv_topk_idx, recv_topk_weights,
                  num_topk, num_scales, hidden_int4, num_recv_buffer_tokens);

    SETUP_LAUNCH_CONFIG(ceil_div(num_tokens * num_ranks, 256), 256, stream);
    LAUNCH_KERNEL(&cfg, fill_int, send_head, -1, num_tokens * num_ranks);
    SETUP_LAUNCH_CONFIG(ceil_div(num_ranks * num_channels, 256), 256, stream);
    LAUNCH_KERNEL(&cfg, fill_int, recv_channel_offset, 0, num_ranks * num_channels);
#else
    (void)recv_x;
    (void)recv_x_scales;
    (void)recv_src_idx;
    (void)recv_topk_idx;
    (void)recv_topk_weights;
    (void)recv_channel_offset;
    (void)send_head;
    (void)x;
    (void)x_scales;
    (void)topk_idx;
    (void)topk_weights;
    (void)is_token_in_rank;
    (void)channel_prefix_matrix;
    (void)num_tokens;
    (void)num_worst_tokens;
    (void)hidden_int4;
    (void)num_topk;
    (void)num_experts;
    (void)num_scales;
    (void)scale_token_stride;
    (void)scale_hidden_stride;
    (void)buffer_ptrs;
    (void)rank;
    (void)num_ranks;
    (void)stream;
    (void)num_sms;
    (void)num_max_send_tokens;
    (void)num_recv_buffer_tokens;
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels,
                           int num_recv_tokens, int num_memset_int,
                           int** barrier_signal_ptrs, int rank, int num_ranks,
                           cudaStream_t stream) {
    (void)buffer_ptrs;
    (void)send_head;
    (void)num_channels;
    (void)num_recv_tokens;
    (void)num_memset_int;
    (void)barrier_signal_ptrs;
    (void)rank;
    (void)num_ranks;
    (void)stream;
}

void combine(cudaDataType_t type,
             void* recv_x, float* recv_topk_weights,
             const void* x, const float* topk_weights,
             const void* bias_0, const void* bias_1,
             const int* src_idx, const int* rank_prefix_matrix, const int* channel_prefix_matrix,
             int* send_head, int num_tokens, int num_recv_tokens, int hidden, int num_topk,
             void** buffer_ptrs, int rank, int num_ranks,
             cudaStream_t stream, int num_sms,
             int num_max_send_tokens, int num_recv_buffer_tokens) {
#ifndef DISABLE_NVSHMEM
    (void)num_sms;
    (void)num_max_send_tokens;
    (void)channel_prefix_matrix;
    (void)send_head;
    const int num_channels = num_sms > 0 ? num_sms / 2 : 1;

    int element_bytes = 0;
    switch (type) {
        case CUDA_R_16BF: element_bytes = sizeof(nv_bfloat16); break;
        case CUDA_R_16F: element_bytes = sizeof(half); break;
        case CUDA_R_32F: element_bytes = sizeof(float); break;
        default: EP_HOST_ASSERT(false and "Unsupported type");
    }
    const int hidden_int4 = static_cast<int>(hidden * element_bytes / sizeof(int4));

    switch (type) {
        case CUDA_R_16BF: {
            SETUP_LAUNCH_CONFIG(ceil_div(num_recv_tokens * hidden, 256), 256, stream);
            LAUNCH_KERNEL(&cfg, init_output<nv_bfloat16>,
                          reinterpret_cast<nv_bfloat16*>(recv_x),
                          reinterpret_cast<const nv_bfloat16*>(bias_0),
                          reinterpret_cast<const nv_bfloat16*>(bias_1),
                          num_recv_tokens, hidden);
            break;
        }
        case CUDA_R_16F: {
            SETUP_LAUNCH_CONFIG(ceil_div(num_recv_tokens * hidden, 256), 256, stream);
            LAUNCH_KERNEL(&cfg, init_output<half>,
                          reinterpret_cast<half*>(recv_x),
                          reinterpret_cast<const half*>(bias_0),
                          reinterpret_cast<const half*>(bias_1),
                          num_recv_tokens, hidden);
            break;
        }
        case CUDA_R_32F: {
            SETUP_LAUNCH_CONFIG(ceil_div(num_recv_tokens * hidden, 256), 256, stream);
            LAUNCH_KERNEL(&cfg, init_output<float>,
                          reinterpret_cast<float*>(recv_x),
                          reinterpret_cast<const float*>(bias_0),
                          reinterpret_cast<const float*>(bias_1),
                          num_recv_tokens, hidden);
            break;
        }
        default:
            EP_HOST_ASSERT(false and "Unsupported type");
    }

    if (recv_topk_weights != nullptr && num_topk > 0) {
        SETUP_LAUNCH_CONFIG(ceil_div(num_recv_tokens * num_topk, 256), 256, stream);
        LAUNCH_KERNEL(&cfg, fill_float, recv_topk_weights, 0.0f, num_recv_tokens * num_topk);
    }

    SETUP_LAUNCH_CONFIG(1, 256, stream);
    LAUNCH_KERNEL(&cfg, clear_channel_head,
                  num_ranks, 0, num_channels, num_recv_buffer_tokens, hidden_int4, num_topk, 0,
                  buffer_ptrs);

    SETUP_LAUNCH_CONFIG(ceil_div(num_recv_buffer_tokens, 256), 256, stream);
    LAUNCH_KERNEL(&cfg, clear_src_idx_buffer,
                  buffer_ptrs, rank, num_ranks, 0, num_channels,
                  num_recv_buffer_tokens, hidden_int4, num_topk, 0);

    SETUP_LAUNCH_CONFIG(ceil_div(num_tokens, 256), 256, stream);
    LAUNCH_KERNEL(&cfg, combine_send,
                  buffer_ptrs, rank, num_ranks, 0, num_channels,
                  reinterpret_cast<const int4*>(x), topk_weights,
                  src_idx, num_tokens, num_topk, hidden_int4, num_recv_buffer_tokens);

    nvshmemx_barrier_all_on_stream(stream);

    switch (type) {
        case CUDA_R_16BF: {
            SETUP_LAUNCH_CONFIG(ceil_div(num_recv_buffer_tokens, 256), 256, stream);
            LAUNCH_KERNEL(&cfg, combine_reduce<nv_bfloat16>,
                          buffer_ptrs, rank, num_ranks, 0, num_channels,
                          reinterpret_cast<nv_bfloat16*>(recv_x), recv_topk_weights,
                          num_recv_tokens, hidden, num_topk, hidden_int4, num_recv_buffer_tokens);
            break;
        }
        case CUDA_R_16F: {
            SETUP_LAUNCH_CONFIG(ceil_div(num_recv_buffer_tokens, 256), 256, stream);
            LAUNCH_KERNEL(&cfg, combine_reduce<half>,
                          buffer_ptrs, rank, num_ranks, 0, num_channels,
                          reinterpret_cast<half*>(recv_x), recv_topk_weights,
                          num_recv_tokens, hidden, num_topk, hidden_int4, num_recv_buffer_tokens);
            break;
        }
        case CUDA_R_32F: {
            SETUP_LAUNCH_CONFIG(ceil_div(num_recv_buffer_tokens, 256), 256, stream);
            LAUNCH_KERNEL(&cfg, combine_reduce<float>,
                          buffer_ptrs, rank, num_ranks, 0, num_channels,
                          reinterpret_cast<float*>(recv_x), recv_topk_weights,
                          num_recv_tokens, hidden, num_topk, hidden_int4, num_recv_buffer_tokens);
            break;
        }
        default:
            EP_HOST_ASSERT(false and "Unsupported type");
    }
#else
    (void)type;
    (void)recv_x;
    (void)recv_topk_weights;
    (void)x;
    (void)topk_weights;
    (void)bias_0;
    (void)bias_1;
    (void)src_idx;
    (void)rank_prefix_matrix;
    (void)channel_prefix_matrix;
    (void)send_head;
    (void)num_tokens;
    (void)num_recv_tokens;
    (void)hidden;
    (void)num_topk;
    (void)buffer_ptrs;
    (void)rank;
    (void)num_ranks;
    (void)stream;
    (void)num_sms;
    (void)num_max_send_tokens;
    (void)num_recv_buffer_tokens;
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

} // namespace intranode

} // namespace deep_ep
