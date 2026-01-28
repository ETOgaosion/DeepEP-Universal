import argparse
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
try:
    from tests.functional_tests.utils import init_dist
except ModuleNotFoundError:
    from utils import init_dist


def _build_all_to_all_layout(num_tokens: int, num_ranks: int, rank: int, device: torch.device):
    token_idx = torch.arange(num_tokens, device=device)
    dst_rank = (token_idx + rank) % num_ranks
    is_token_in_rank = torch.zeros((num_tokens, num_ranks), dtype=torch.bool, device=device)
    is_token_in_rank.scatter_(1, dst_rank.view(-1, 1), True)
    num_tokens_per_rank = is_token_in_rank.to(torch.int32).sum(dim=0, dtype=torch.int32).contiguous()
    num_tokens_per_expert = num_tokens_per_rank.clone().contiguous()
    return is_token_in_rank.contiguous(), num_tokens_per_rank, num_tokens_per_expert


def _check_recv_by_rank(recv_x: torch.Tensor, rank_prefix_matrix: torch.Tensor, rank: int, num_ranks: int):
    check_start = 0
    for src_rank in range(num_ranks):
        check_end = int(rank_prefix_matrix[src_rank][rank].item())
        if check_end > check_start:
            segment = recv_x[check_start:check_end]
            expected = torch.full_like(segment, src_rank)
            if not torch.equal(segment, expected):
                raise AssertionError(
                    f"recv_x mismatch for src_rank={src_rank} rank={rank}: "
                    f"range=({check_start}, {check_end})"
                )
        check_start = check_end


def _resolve_p2p_chunked_tokens(args: argparse.Namespace, num_tokens: int) -> tuple[int, int]:
    send_tokens = args.num_max_p2p_chunked_send_tokens or 6
    recv_tokens = args.num_max_p2p_chunked_recv_tokens or max(256, num_tokens)
    if recv_tokens <= send_tokens:
        recv_tokens = send_tokens + 1
    return send_tokens, recv_tokens


def _compute_p2p_bytes(args: argparse.Namespace, hidden: int, num_ranks: int, num_tokens: int) -> int:
    send_tokens, recv_tokens = _resolve_p2p_chunked_tokens(args, num_tokens)
    config = deep_ep.Config(
        args.num_sms,
        send_tokens,
        recv_tokens,
        1,
        2,
    )
    hidden_bytes = hidden * torch.tensor([], dtype=torch.bfloat16).element_size()
    return max(args.p2p_bytes, int(config.get_nvl_buffer_size_hint(hidden_bytes, num_ranks)))


def _run_all_to_all(args: argparse.Namespace, local_rank: int, num_ranks: int, rank: int,
                    buffer: deep_ep.Buffer, group: dist.ProcessGroup):
    num_tokens = args.num_tokens
    hidden = args.hidden
    device = torch.device("cuda", local_rank)

    x = torch.full((num_tokens, hidden), float(rank), dtype=torch.bfloat16, device=device)
    is_token_in_rank, num_tokens_per_rank, num_tokens_per_expert = _build_all_to_all_layout(
        num_tokens, num_ranks, rank, device
    )

    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    send_tokens, recv_tokens = _resolve_p2p_chunked_tokens(args, num_tokens)
    config = deep_ep.Config(
        args.num_sms,
        send_tokens,
        recv_tokens,
        1,
        2,
    )

    recv_x, _, _, _, handle, event = buffer.dispatch(
        x,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
        async_finish=False,
    )
    if getattr(event, "event", None) is not None:
        event.current_stream_wait()

    expected_recv = int(gbl_num_tokens_per_rank[rank].item())
    if recv_x.size(0) != expected_recv:
        raise AssertionError(f"recv_x.size(0)={recv_x.size(0)} expected={expected_recv}")
    else:
        if local_rank == 0:
            print(f"[success] rank={rank} recv_x.size(0)={recv_x.size(0)} as expected", flush=True)

    rank_prefix_matrix = handle[0]
    _check_recv_by_rank(recv_x, rank_prefix_matrix, rank, num_ranks)

    combined_x, _, event = buffer.combine(recv_x, handle, config=config, async_finish=False)
    if getattr(event, "event", None) is not None:
        event.current_stream_wait()

    print(f"combined_x.size={combined_x.size()} combined_x.dtype={combined_x.dtype} combined_x.device={combined_x.device}", flush=True)
    print(f"x.size={x.size()} x.dtype={x.dtype} x.device={x.device}", flush=True)
    if not torch.equal(combined_x.to("cpu"), x.to("cpu")):
        raise AssertionError("combined_x does not match original x")
    else:
        if local_rank == 0:
            print(f"[success] rank={rank} combined_x matches original x", flush=True)

    if local_rank == 0:
        print("[ok] intranode P2P all-to-all dispatch + combine passed", flush=True)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    p2p_bytes = _compute_p2p_bytes(args, args.hidden, num_ranks, args.num_tokens)

    buffer = deep_ep.Buffer(
        group,
        p2p_bytes,
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )

    _run_all_to_all(args, local_rank, num_ranks, rank, buffer, group)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal intranode CUDA P2P all-to-all test")
    parser.add_argument("--num-processes", type=int, default=2,
                        help="Number of processes to spawn (default: 2)")
    parser.add_argument("--num-tokens", type=int, default=1024,
                        help="Number of tokens per rank (default: 1024)")
    parser.add_argument("--hidden", type=int, default=1024,
                        help="Hidden dimension size (default: 1024, must be divisible by 8)")
    parser.add_argument("--num-sms", type=int, default=20,
                        help="Number of SMs for intranode kernels (default: 20)")
    parser.add_argument("--num-max-p2p-chunked-send-tokens", type=int, default=6,
                        help="Max P2P chunked send tokens (default: 6)")
    parser.add_argument("--num-max-p2p-chunked-recv-tokens", type=int, default=0,
                        help="Max P2P chunked recv tokens (default: auto)")
    parser.add_argument("--p2p-bytes", type=int, default=0,
                        help="Override P2P buffer bytes; 0 uses the computed minimum")
    args = parser.parse_args()

    if args.hidden % 8 != 0:
        raise ValueError("--hidden must be divisible by 8 for int4 alignment")

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
