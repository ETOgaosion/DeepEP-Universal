# Intranode NVSHMEM check against main NVSHMEM 3.5 fixes

Date: 2026-01-28

## Reference
- Main commit: `29d31c0` ("Nvshmem 3 5 runtime fixes")
  - Forces CUDA RDC when NVSHMEM is enabled.
  - Updates IBGDA RC layout access for NVSHMEM 3.5.

## Findings
- `csrc/kernels/intranode_nvshmem.cu` already includes `configs.cuh`, so it inherits the NVSHMEM 3.5 RDC fix; it does not use IBGDA RC symbols, so the RC layout fix is not directly applicable.
- In intranode NVSHMEM mode, `Buffer::intranode_dispatch`/`Buffer::intranode_combine` were still passing IPC `buffer_ptrs_gpu` into the intranode NVSHMEM kernels. Those kernels expect NVSHMEM symmetric memory. This would have used non-symmetric addresses.
- `deep_ep/buffer.py` had an indentation error that prevented NVSHMEM env tuning from executing (and would raise an `IndentationError`).

## Actions taken
- Switched intranode NVSHMEM kernel calls to use `intranode_nvshmem_buffer_ptrs_gpu` (and null barrier pointers) when `use_nvshmem_intranode` is enabled.
- Fixed indentation in `deep_ep/buffer.py` so the NVSHMEM env block runs correctly.

## Files updated
- `csrc/deep_ep.cpp`
- `deep_ep/buffer.py`
