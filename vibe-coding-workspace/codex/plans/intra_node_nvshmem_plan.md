# Intra Node NVSHMEM Plan

## Goal
Replace CUDA P2P intranode all-to-all with NVSHMEM-based intranode dispatch/combine, remove NVLink assumptions, and add a functional test.

## Constraints & Notes
- Use NVSHMEM APIs for intra-node communication; assume no NVLink.
- Support SM80/SM90/SM120 without architecture-specific intrinsics.
- Preserve existing public C++/Python API signatures.
- Keep buffer size constraints compatible with existing `Config.get_nvl_buffer_size_hint`.

## Approach
1. **NVSHMEM buffer setup**
   - Initialize NVSHMEM even when `num_rdma_bytes == 0` but `num_nvl_bytes > 0`.
   - Allocate a symmetric NVSHMEM buffer for intranode (`num_nvl_bytes`).
   - Provide a device-side pointer array for intranode kernels.
   - Skip CUDA IPC/NVLink checks for intranode-only mode.

2. **NVSHMEM intranode kernels (simplified)**
   - Keep existing API functions: `notify_dispatch`, `cached_notify_dispatch`, `dispatch`, `cached_notify_combine`, `combine`.
   - Use NVSHMEM put/get to exchange data via a shared NVSHMEM buffer.
   - Use a single logical channel (channel 0) but keep the legacy layout and buffer size assumptions.
   - Add explicit NVSHMEM stream barriers between send and local copy/reduction phases.

3. **Dispatch flow**
   - Write local counts into NVSHMEM buffer; barrier.
   - Gather counts from all ranks via NVSHMEM get into `rank_prefix_matrix` and compute local expert totals.
   - Send tokens to destination buffers with NVSHMEM puts (slot allocation via local counters).
   - Barrier, then locally copy NVSHMEM buffer into output tensors.

4. **Combine flow**
   - Send contributions back to original ranks using NVSHMEM puts into per-rank buffers.
   - Barrier, then locally reduce contributions into output (atomic adds) and apply bias if provided.

5. **Testing**
   - Add `tests/functional_tests/test_intranode_nvshmem.py` modeled after existing intranode test.
   - Ensure it uses the same public API and validates round-trip correctness.

## Risks / Follow-ups
- Combine uses atomic adds on CUDA types; ensure supported for float/half/bfloat16 on SM80+.
- For workloads where `num_recv_tokens > num_max_nvl_chunked_recv_tokens`, add chunking or document constraints.
