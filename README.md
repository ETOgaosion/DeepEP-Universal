# DeepEP-Universal

DeepEP is a communication library tailored for Mixture-of-Experts (MoE) and expert parallelism (EP). It provides high-throughput and low-latency all-to-all GPU kernels, which are also known as MoE dispatch and combine. The library also supports low-precision operations, including FP8.

DeepEP-Universal is a library trying to use deepep in all common GPUs, including RTX 5090, or machines without IBGDA.

To align with the group-limited gating algorithm proposed in the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper, DeepEP offers a set of kernels optimized for asymmetric-domain bandwidth forwarding, such as forwarding data from NVLink domain to RDMA domain. These kernels deliver high throughput, making them suitable for both training and inference prefilling tasks. Additionally, they support SM (Streaming Multiprocessors) number control.

For latency-sensitive inference decoding, DeepEP includes a set of low-latency kernels with pure RDMA to minimize delays. The library also introduces a hook-based communication-computation overlapping method that does not occupy any SM resource.

Notice: the implementation in this library may have some slight differences from the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper.

## Quick start

### Requirements

- Ampere (SM80), Hopper (SM90) GPUs, Blackwell(SM120) GPUs
- Python 3.8 and above
- CUDA version
  - CUDA 12.8 tested
- PyTorch 2.8.0+cu128 tested
- NVSHMEM support

To use 5090, you should enable P2P access, follow [this repo](https://github.com/aikitoria/open-gpu-kernel-modules/tree/580.82.09-p2p)

### Download and install NVSHMEM dependency

See [third-party/nvshmem](./third-party/)

### Docker

Use whatcanyousee/deepep-universal:v0.1-pytorch25.02 , nvshmem is already installed

### Installation

```bash
source scripts/env.sh
TORCH_CUDA_ARCH_LIST=xxx python setup.py install

# RTX 5090
DISABLE_AGGRESSIVE_PTX_INSTRS=1 TORCH_CUDA_ARCH_LIST="12.0" python setup.py install
```

#### Installation environment variables

- `NVSHMEM_DIR`: the path to the NVSHMEM directory, disable all internode and low-latency features if not specified
- `DISABLE_SM90_FEATURES`: 0 or 1, whether to disable SM90 features, it is required for SM90 devices or CUDA 11
- `TORCH_CUDA_ARCH_LIST`: the list of target architectures, e.g. `TORCH_CUDA_ARCH_LIST="9.0"`
- `DISABLE_AGGRESSIVE_PTX_INSTRS`: 0 or 1, whether to disable aggressive load/store instructions, see [Undefined-behavior PTX usage](#undefined-behavior-ptx-usage) for more details

Then, import `deep_ep` in your Python project, and enjoy!

### Test intranode (NVSHMEM)

Run: `python tests/functional_tests/test_intranode_nvshmem.py`

## Network configurations

DeepEP is fully tested with InfiniBand networks. However, it is theoretically compatible with RDMA over Converged Ethernet (RoCE) as well.

### Traffic isolation

Traffic isolation is supported by InfiniBand through Virtual Lanes (VL).

To prevent interference between different types of traffic, we recommend segregating workloads across different virtual lanes as follows:

- workloads using normal kernels
- workloads using low-latency kernels
- other workloads

For DeepEP, you can control the virtual lane assignment by setting the `NVSHMEM_IB_SL` environment variable.

### Adaptive routing

Adaptive routing is an advanced routing feature provided by InfiniBand switches that can evenly distribute traffic across multiple paths. Enabling adaptive routing can completely eliminate network congestion caused by routing conflicts, but it also introduces additional latency. We recommend the following configuration for optimal performance:

- enable adaptive routing in environments with heavy network loads
- use static routing in environments with light network loads

### Congestion control

Congestion control is disabled as we have not observed significant congestion in our production environment.

## Interfaces and examples

### Example use in model training

### Example: intranode NVSHMEM all-to-all

Use the explicit NVSHMEM APIs (`dispatch_nvshmem` / `combine_nvshmem`) to ensure the intranode NVSHMEM path is used.

```python
import torch
import torch.distributed as dist
from deep_ep import Buffer, Config

def run_intranode_nvshmem(group: dist.ProcessGroup, num_tokens: int, hidden: int):
    rank = group.rank()
    num_ranks = group.size()
    device = torch.device("cuda", rank)

    x = torch.full((num_tokens, hidden), float(rank), dtype=torch.bfloat16, device=device)
    token_idx = torch.arange(num_tokens, device=device)
    dst_rank = (token_idx + rank) % num_ranks
    is_token_in_rank = torch.zeros((num_tokens, num_ranks), dtype=torch.bool, device=device)
    is_token_in_rank.scatter_(1, dst_rank.view(-1, 1), True)
    num_tokens_per_rank = is_token_in_rank.to(torch.int32).sum(dim=0, dtype=torch.int32).contiguous()
    num_tokens_per_expert = num_tokens_per_rank.clone().contiguous()

    # NVSHMEM-only intranode: num_nvl_bytes > 0, num_rdma_bytes == 0
    config = Config(num_sms=20,
                    num_max_nvl_chunked_send_tokens=6,
                    num_max_nvl_chunked_recv_tokens=max(256, num_tokens),
                    num_max_rdma_chunked_send_tokens=1,
                    num_max_rdma_chunked_recv_tokens=2)
    hidden_bytes = hidden * torch.tensor([], dtype=torch.bfloat16).element_size()
    num_nvl_bytes = int(config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks))
    buffer = Buffer(group, num_nvl_bytes, 0)

    recv_x, _, _, _, handle, _ = buffer.dispatch_nvshmem(
        x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
    )
    combined_x, _, _ = buffer.combine_nvshmem(recv_x, handle, config=config)
    return combined_x
```

The normal kernels can be used in model training or the inference prefilling phase (without the backward part) as the below example code shows.

```python
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Union

from deep_ep import Buffer, EventOverlap

# Communication buffer (will allocate at runtime)
_buffer: Optional[Buffer] = None

# Set the number of SMs to use
# NOTES: this is a static variable
Buffer.set_num_sms(24)


# You may call this function at the framework initialization
def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> Buffer:
    global _buffer

    # NOTES: you may also replace `get_*_config` with your auto-tuned results via all the tests
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (Buffer.get_dispatch_config(group.size()), Buffer.get_combine_config(group.size())):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    # Allocate a buffer if not existed or not enough buffer size
    if _buffer is None or _buffer.group != group or _buffer.num_nvl_bytes < num_nvl_bytes or _buffer.num_rdma_bytes < num_rdma_bytes:
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def dispatch_forward(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                     num_experts: int, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, List, Tuple, EventOverlap]:
    # NOTES: an optional `previous_event` means a CUDA event captured that you want to make it as a dependency
    # of the dispatch kernel, it may be useful with communication-computation overlap. For more information, please
    # refer to the docs of `Buffer.dispatch`
    global _buffer

    # Calculate layout before actual dispatch
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(topk_idx, num_experts,
                                    previous_event=previous_event, async_finish=True,
                                    allocate_on_comm_stream=previous_event is not None)
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph
    # Unless you specify `num_worst_tokens`, but this flag is for intranode only
    # For more advanced usages, please refer to the docs of the `dispatch` function
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=True,
                         allocate_on_comm_stream=True)
    # For event management, please refer to the docs of the `EventOverlap` class
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event


def dispatch_backward(grad_recv_x: torch.Tensor, grad_recv_topk_weights: torch.Tensor, handle: Tuple) -> \
        Tuple[torch.Tensor, torch.Tensor, EventOverlap]:
    global _buffer

    # The backward process of MoE dispatch is actually a combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_grad_x, combined_grad_recv_topk_weights, event = \
        _buffer.combine(grad_recv_x, handle, topk_weights=grad_recv_topk_weights, async_finish=True)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_grad_x, combined_grad_recv_topk_weights, event


def combine_forward(x: torch.Tensor, handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[torch.Tensor, EventOverlap]:
    global _buffer

    # Do MoE combine
    # For more advanced usages, please refer to the docs of the `combine` function
    combined_x, _, event = _buffer.combine(x, handle, async_finish=True, previous_event=previous_event,
                                           allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return combined_x, event


def combine_backward(grad_combined_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     handle: Tuple, previous_event: Optional[EventOverlap] = None) -> \
        Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], EventOverlap]:
    global _buffer

    # The backward process of MoE combine is actually a dispatch
    # For more advanced usages, please refer to the docs of the `dispatch` function
    grad_x, _, _, _, _, event = _buffer.dispatch(grad_combined_x, handle=handle, async_finish=True,
                                                 previous_event=previous_event,
                                                 allocate_on_comm_stream=previous_event is not None)

    # For event management, please refer to the docs of the `EventOverlap` class
    return grad_x, event
```

Moreover, inside the dispatch function, we may not know how many tokens to receive for the current rank. So an implicit CPU wait for GPU received count signal will be involved, as the following figure shows.

![normal](figures/normal.png)

![low-latency](figures/low-latency.png)

## Roadmap

- [x] AR support
- [x] Refactor low-latency mode AR code
- [x] A100 support (intranode only)
- [x] Support BF16 for the low-latency dispatch kernel
- [x] Support NVLink protocol for intranode low-latency kernels
- [ ] TMA copy instead of LD/ST
  - [x] Intranode kernels
  - [ ] Internode kernels
  - [ ] Low-latency kernels
- [ ] SM-free kernels and refactors
- [ ] Fully remove undefined-behavior PTX instructions

## Notices

#### Easier potential overall design

The current DeepEP implementation uses queues for communication buffers which save memory but introduce complexity and potential deadlocks. If you're implementing your own version based on DeepEP, consider using fixed-size buffers allocated to maximum capacity for simplicity and better performance. For a detailed discussion of this alternative approach, see https://github.com/deepseek-ai/DeepEP/issues/39.

#### Undefined-behavior PTX usage

- For extreme performance, we discover and use an undefined-behavior PTX usage: using read-only PTX `ld.global.nc.L1::no_allocate.L2::256B` to **read volatile data**. The PTX modifier `.nc` indicates that a non-coherent cache is used. But the correctness is tested to be guaranteed with `.L1::no_allocate` on Hopper architectures, and performance will be much better. The reason we guess may be: the non-coherent cache is unified with L1, and the L1 modifier is not just a hint but a strong option, so that the correctness can be guaranteed by no dirty data in L1.
- Initially, because NVCC could not automatically unroll volatile read PTX, we tried using `__ldg` (i.e., `ld.nc`). Even compared to manually unrolled volatile reads, it was significantly faster (likely due to additional compiler optimizations). However, the results could be incorrect or dirty. After consulting the PTX documentation, we discovered that L1 and non-coherent cache are unified on Hopper architectures. We speculated that `.L1::no_allocate` might resolve the issue, leading to this discovery.
- If you find kernels not working on some other platforms, you may add `DISABLE_AGGRESSIVE_PTX_INSTRS=1` to `setup.py` and disable this, or file an issue.

#### Auto-tuning on your cluster

For better performance on your cluster, we recommend to run all the tests and use the best auto-tuned configuration. The default configurations are optimized on the DeepSeek's internal cluster.

## License

This code repository is released under [the MIT License](LICENSE), except for codes that reference NVSHMEM (including `csrc/kernels/ibgda_device.cuh` and `third-party/nvshmem.patch`), which are subject to [NVSHMEM SLA](https://docs.nvidia.com/nvshmem/api/sla.html).

## Experimental Branches

- [Zero-copy](https://github.com/deepseek-ai/DeepEP/pull/453)
  - Removing the copy between PyTorch tensors and communication buffers, which reduces the SM usages significantly for normal kernels
  - This PR is authored by **Tencent Network Platform Department**
- [Eager](https://github.com/deepseek-ai/DeepEP/pull/437)
  - Using a low-latency protocol removes the extra RTT latency introduced by RDMA atomic OPs
- [Hybrid-EP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
  - A new backend implementation using TMA instructions for minimal SM usage and larger NVLink domain support
  - Fine-grained communication-computation overlap for single-batch scenarios
  - PCIe kernel support for non-NVLink environments
  - NVFP4 data type support
- [AntGroup-Opt](https://github.com/deepseek-ai/DeepEP/tree/antgroup-opt)
  - This optimization series is authored by **AntGroup Network Platform Department**
  - [Normal-SMFree](https://github.com/deepseek-ai/DeepEP/pull/347) Eliminating SM from RDMA path by decoupling comm-kernel execution from NIC token transfer, freeing SMs for compute
  - [LL-SBO](https://github.com/deepseek-ai/DeepEP/pull/483) Overlapping Down GEMM computation with Combine Send communication via signaling mechanism to reduce end-to-end latency
  - [LL-Layered](https://github.com/deepseek-ai/DeepEP/pull/500) Optimizing cross-node LL operator communication using rail-optimized forwarding and data merging to reduce latency

## Community Forks

- [uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep) - Enables running DeepEP on heterogeneous GPUs (e.g., Nvidia, AMD) and NICs (e.g., EFA, Broadcom, CX7)
- [Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP) - Adds multi-QP solution and dual-port NIC support in IBRC transport
- [antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace) - A diagnostic analyzer for efficient and precise localization of slow ranks

## Citation

If you use this codebase or otherwise find our work valuable, please cite:

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```
