#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import sys
import datetime
from pathlib import Path

import torch
import torch.distributed as dist

try:
    from pssh.clients import ParallelSSHClient
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "parallel-ssh is required. Install with: pip install parallel-ssh"
    ) from exc


def _load_env_config(repo_root: Path):
    env_path = repo_root / ".secrets" / "env.json"
    if not env_path.exists():
        raise SystemExit(f"Missing env config: {env_path}")

    return json.loads(env_path.read_text(encoding="utf-8"))


def _build_env_exports(env_items):
    parts = []
    for key, value in env_items.items():
        if value is None:
            continue
        parts.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(parts)


def _run_worker(args):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    timeout = datetime.timedelta(seconds=args.timeout)

    print(
        f"[rank {args.rank}] init_process_group backend={args.backend} "
        f"master={args.master_addr}:{args.master_port} world_size={args.world_size}"
    )

    dist.init_process_group(
        backend=args.backend,
        rank=args.rank,
        world_size=args.world_size,
        timeout=timeout,
    )

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"[rank {args.rank}] cuda requested but not available", file=sys.stderr)
            return 3

        tensor = torch.tensor([1], device=device, dtype=torch.int32) * (args.rank + 1)
        dist.barrier()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()

        print(f"[rank {args.rank}] all_reduce result={tensor.item()}")

        if args.world_size == 2:
            expected = 1 + 2
            if tensor.item() != expected:
                print(
                    f"[rank {args.rank}] unexpected result: {tensor.item()} "
                    f"(expected {expected})",
                    file=sys.stderr,
                )
                return 4
    finally:
        dist.destroy_process_group()

    return 0


def _run_controller(args):
    repo_root = Path(__file__).resolve().parents[3]
    cfg = _load_env_config(repo_root)

    master_addr = cfg.get("MASTER_NODE_ADDR")
    slave_addrs = list(cfg.get("SLAVE_NODE_ADDRS", []))
    if not master_addr:
        raise SystemExit("MASTER_NODE_ADDR is missing in .secrets/env.json")
    if not slave_addrs:
        raise SystemExit("SLAVE_NODE_ADDRS is empty in .secrets/env.json")

    hosts = [master_addr] + slave_addrs
    nnodes = args.nnodes or len(hosts)
    if nnodes != len(hosts):
        raise SystemExit(
            f"--nnodes ({nnodes}) does not match host list length ({len(hosts)})."
        )
    nproc_per_node = args.nproc_per_node
    if nproc_per_node <= 0:
        raise SystemExit("--nproc-per-node is required and must be > 0.")
    world_size = args.world_size if args.world_size > 0 else nnodes * nproc_per_node
    if args.world_size > 0 and world_size != nnodes * nproc_per_node:
        raise SystemExit(
            f"--world-size ({args.world_size}) must equal nnodes*nproc-per-node "
            f"({nnodes}*{nproc_per_node}={nnodes * nproc_per_node})."
        )

    client_kwargs = {}
    if args.user:
        client_kwargs["user"] = args.user
    if args.identity_file:
        client_kwargs["pkey"] = args.identity_file
    if args.ssh_port:
        client_kwargs["port"] = args.ssh_port

    python_bin = os.getenv("PYTHON") or "python"
    conda_sh = os.getenv("CONDA_SH", "$HOME/miniconda3/etc/profile.d/conda.sh")
    conda_env = os.getenv("CONDA_ENV", "deepep")
    env_sh = os.getenv("ENV_SH", "")
    env_sh_cmd = f"source {shlex.quote(env_sh)} && " if env_sh else ""

    commands = []
    for node_rank, host in enumerate(hosts):
        env_items = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": args.master_port,
            "WORLD_SIZE": world_size,
            "NNODES": nnodes,
            "NPROC_PER_NODE": nproc_per_node,
            "NODE_RANK": node_rank,
            "DIST_BACKEND": args.backend,
            "DIST_TIMEOUT": args.timeout,
            "DIST_DEVICE": args.device,
        }
        env_prefix = _build_env_exports(env_items)
        cmd = (
            f"cd {shlex.quote(str(repo_root))} && "
            f"source {conda_sh} && conda activate {shlex.quote(conda_env)} && "
            f"{env_sh_cmd}"
            f"{env_prefix} {shlex.quote(python_bin)} -m torch.distributed.run "
            f"--nnodes {nnodes} --nproc_per_node {nproc_per_node} "
            f"--node_rank {node_rank} "
            f"--master_addr {shlex.quote(master_addr)} --master_port {shlex.quote(str(args.master_port))} "
            f"tests/trial/test_multinodes/dist_two_nodes_ping.py --worker"
        ).strip()
        commands.append(cmd)

    client = ParallelSSHClient(hosts, **client_kwargs)
    for host, cmd in zip(hosts, commands):
        print(f"[{host}] cmd: {cmd}")

    output = client.run_command(
        "%(cmd)s",
        host_args=[{"cmd": cmd} for cmd in commands],
        stop_on_errors=False,
    )
    client.join(output)

    exit_code = 0
    for host, host_output in zip(hosts, output):
        if host_output is None:
            exit_code = exit_code or 1
            print(f"[{host}] no output object returned", file=sys.stderr)
            continue
        if getattr(host_output, "exception", None):
            exit_code = exit_code or 1
            print(f"[{host}] exception: {host_output.exception}", file=sys.stderr)
        for line in (host_output.stdout or []):
            print(f"[{host}] {line}")
        for line in (host_output.stderr or []):
            print(f"[{host}][stderr] {line}", file=sys.stderr)
        if host_output.exit_code != 0:
            exit_code = host_output.exit_code
            print(f"[{host}] exit code: {host_output.exit_code}", file=sys.stderr)

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Simple 2-node torch.distributed connectivity check.")
    parser.add_argument("--worker", action="store_true", help="Run as worker (do not launch parallel-ssh).")
    parser.add_argument("--backend", default=os.getenv("DIST_BACKEND", "nccl"), help="Process group backend.")
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", "-1")), help="Global rank.")
    parser.add_argument("--world-size", type=int, default=int(os.getenv("WORLD_SIZE", "-1")), help="World size.")
    parser.add_argument("--nnodes", type=int, default=int(os.getenv("NNODES", "0")), help="Number of nodes.")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=int(os.getenv("NPROC_PER_NODE", "0")),
        help="Number of processes per node.",
    )
    parser.add_argument("--master-addr", default=os.getenv("MASTER_ADDR", ""), help="Master address.")
    parser.add_argument("--master-port", default=os.getenv("MASTER_PORT", "29500"), help="Master port.")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("DIST_TIMEOUT", "120")), help="Init timeout seconds.")
    parser.add_argument("--device", default=os.getenv("DIST_DEVICE", "cpu"), help="cpu or cuda.")
    parser.add_argument("--user", default=os.getenv("SSH_USER"), help="SSH user (optional).")
    parser.add_argument("--identity-file", default=os.getenv("SSH_IDENTITY_FILE"), help="SSH identity file (optional).")
    parser.add_argument("--ssh-port", type=int, default=int(os.getenv("SSH_PORT", "22")), help="SSH port.")
    args = parser.parse_args()

    if args.worker:
        if args.rank < 0 or args.world_size <= 0:
            print("RANK and WORLD_SIZE must be set (env or flags).", file=sys.stderr)
            return 2
        if not args.master_addr:
            print("MASTER_ADDR must be set (env or flag).", file=sys.stderr)
            return 2
        return _run_worker(args)

    return _run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
