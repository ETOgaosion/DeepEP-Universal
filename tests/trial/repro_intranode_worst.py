import argparse
import os
import sys
import time
import torch
import torch.distributed as dist

# Ensure repo root on sys.path for tests.functional_tests imports
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# noinspection PyUnresolvedReferences
import deep_ep

from tests.functional_tests.utils import init_dist, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Torch API compatibility shim for older builds
if not hasattr(torch.cuda, "device_can_access_peer"):
    if hasattr(torch.cuda, "can_device_access_peer"):
        torch.cuda.device_can_access_peer = torch.cuda.can_device_access_peer  # type: ignore[attr-defined]
    elif hasattr(torch.cuda, "_can_device_access_peer"):
        torch.cuda.device_can_access_peer = torch.cuda._can_device_access_peer  # type: ignore[attr-defined]
    else:
        def _assume_peer_access(_src: int, _dst: int) -> bool:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    print("[warn] torch.cuda.device_can_access_peer not available; skipping P2P check", flush=True)
            return True
        torch.cuda.device_can_access_peer = _assume_peer_access  # type: ignore[attr-defined]


def sync(msg: str):
    torch.cuda.synchronize()
    if torch.distributed.get_rank() == 0:
        print(f"[sync] {msg}", flush=True)


def run_once(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int,
             buffer: deep_ep.Buffer, group: dist.ProcessGroup):
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()

    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0

    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

    current_x = x_e4m3 if args.fp8 else x
    dispatch_args = {
        'x': current_x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'config': config,
        'async_finish': False,
        'topk_idx': topk_idx,
        'topk_weights': topk_weights,
    }

    sync("before dispatch")
    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
    if getattr(event, "event", None) is not None:
        event.current_stream_wait()
    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
    sync("after dispatch")

    if local_rank == 0:
        print("[ok] standard dispatch path passed", flush=True)


# noinspection PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(group, int(2e9), 0, low_latency_mode=False, num_qps_per_rank=1, explicitly_destroy=True)
    torch.manual_seed(rank)

    for num_sms in (24,):
        run_once(args, num_sms, local_rank, num_ranks, rank, buffer, group)
        if local_rank == 0:
            print('', flush=True)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Repro for intranode worst-tokens path')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-tokens', type=int, default=4096)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--num-topk', type=int, default=8)
    parser.add_argument('--num-experts', type=int, default=256)
    parser.add_argument('--fp8', action='store_true')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
