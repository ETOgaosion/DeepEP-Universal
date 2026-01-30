#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import sys
import datetime
from pathlib import Path
import inspect
import socket

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

def _filter_kwargs(func, kwargs):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def _parse_kv(items):
    if not items:
        return {}
    result = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --extra-env value '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"Invalid --extra-env value '{item}', empty key")
        result[key] = value
    return result


def _run_worker(args):
    log_dir = Path(args.log_dir or "tests/trial/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname()
    log_path = log_dir / f"dist_two_nodes_ping_rank{args.rank}_{hostname}.log"
    log_fp = log_path.open("a", buffering=1, encoding="utf-8")

    def _log(msg):
        log_fp.write(msg + "\n")
        log_fp.flush()

    _log(f"[rank {args.rank}] logging to {log_path}")
    _log(
        f"[rank {args.rank}] env RANK={os.getenv('RANK')} "
        f"WORLD_SIZE={os.getenv('WORLD_SIZE')} "
        f"LOCAL_RANK={os.getenv('LOCAL_RANK')} "
        f"NNODES={os.getenv('NNODES')} "
        f"NPROC_PER_NODE={os.getenv('NPROC_PER_NODE')}"
    )
    try:
        resolved = socket.gethostbyname(args.master_addr)
        _log(f"[rank {args.rank}] master_addr={args.master_addr} resolved={resolved}")
        if resolved.startswith("127."):
            _log(f"[rank {args.rank}] WARNING: master_addr resolves to loopback")
    except Exception as exc:
        _log(f"[rank {args.rank}] WARNING: failed to resolve master_addr: {exc}")

    try:
        nnodes_env = int(os.getenv("NNODES", "0"))
        nproc_env = int(os.getenv("NPROC_PER_NODE", "0"))
        if nnodes_env > 0 and nproc_env > 0:
            expected_world = nnodes_env * nproc_env
            if args.world_size != expected_world:
                _log(
                    f"[rank {args.rank}] ERROR: WORLD_SIZE={args.world_size} "
                    f"!= NNODES*NPROC_PER_NODE ({nnodes_env}*{nproc_env}={expected_world})"
                )
                return 5
    except Exception as exc:
        _log(f"[rank {args.rank}] WARNING: failed to validate world size: {exc}")

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    timeout = datetime.timedelta(seconds=args.timeout)

    device = torch.device(args.device)
    local_rank = None
    if device.type == "cuda":
        local_rank = args.local_rank
        if local_rank is None or local_rank < 0:
            if args.nproc_per_node and args.nproc_per_node > 0:
                local_rank = args.rank % args.nproc_per_node
            else:
                local_rank_env = os.getenv("LOCAL_RANK")
                if local_rank_env is not None:
                    local_rank = int(local_rank_env)
        if local_rank is not None and local_rank >= 0:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        _log(f"[rank {args.rank}] using cuda device {device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        _log(f"[rank {args.rank}] cuda requested but not available")
        return 3

    _log(
        f"[rank {args.rank}] init_process_group backend={args.backend} "
        f"master={args.master_addr}:{args.master_port} world_size={args.world_size}"
    )

    init_kwargs = dict(
        backend=args.backend,
        rank=args.rank,
        world_size=args.world_size,
        timeout=timeout,
    )
    if device.type == "cuda" and local_rank is not None and local_rank >= 0:
        try:
            sig = inspect.signature(dist.init_process_group)
            if "device_id" in sig.parameters:
                init_kwargs["device_id"] = local_rank
        except (TypeError, ValueError):
            pass
    
    _log(f"[rank {args.rank}] init_process_group kwargs: {init_kwargs}")
    dist.init_process_group(**init_kwargs)
    
    _log(f"[rank {args.rank}] init_process_group complete")

    try:
        tensor = torch.arange(2, device=device, dtype=torch.int64) + 1 + args.world_size * args.rank
        _log(f"[rank {args.rank}] initial tensor={tensor}")
        dist.barrier()
        _log(f"[rank {args.rank}] passed barrier")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        _log(f"[rank {args.rank}] all_reduce complete")
        dist.barrier()

        _log(f"[rank {args.rank}] all_reduce result={tensor}")
    finally:
        dist.destroy_process_group()
        log_fp.close()
        if log_fp is not None:
            log_fp.close()

    return 0


def _run_controller(args):
    repo_root = Path(__file__).resolve().parents[3]
    cfg = _load_env_config(repo_root)

    ssh_master_addr = cfg.get("SSH_MASTER_NODE_ADDR") or cfg.get("MASTER_NODE_ADDR")
    ssh_slave_addrs = list(cfg.get("SSH_SLAVE_NODE_ADDRS", [])) or list(cfg.get("SLAVE_NODE_ADDRS", []))
    master_addr = cfg.get("MASTER_NODE_IP")
    if not ssh_master_addr:
        raise SystemExit("SSH_MASTER_NODE_ADDR (or MASTER_NODE_ADDR) is missing in .secrets/env.json")
    if not ssh_slave_addrs:
        raise SystemExit("SSH_SLAVE_NODE_ADDRS (or SLAVE_NODE_ADDRS) is empty in .secrets/env.json")
    if not master_addr:
        raise SystemExit("MASTER_NODE_IP is missing in .secrets/env.json")

    hosts = [ssh_master_addr] + ssh_slave_addrs
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
    if args.timeout:
        client_kwargs["timeout"] = args.timeout

    python_bin = os.getenv("PYTHON") or "python"
    conda_sh = os.getenv("CONDA_SH", "$HOME/miniconda3/etc/profile.d/conda.sh")
    conda_env = os.getenv("CONDA_ENV", "deepep")
    env_sh = os.getenv("ENV_SH", "")
    env_sh_cmd = f"source {shlex.quote(env_sh)} && " if env_sh else ""

    extra_env = {}
    if isinstance(cfg.get("EXTRA_ENVS"), dict):
        extra_env.update(cfg.get("EXTRA_ENVS"))
    test_envs = cfg.get("TEST_ENVS", {})
    if isinstance(test_envs, dict) and isinstance(test_envs.get("dist_two_nodes_ping"), dict):
        extra_env.update(test_envs.get("dist_two_nodes_ping"))
    extra_env.update(_parse_kv(args.extra_env))
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
            "LOG_DIR": args.log_dir,
            "PYTHONUNBUFFERED": "1",
        }
        env_items = {**env_items, **extra_env}
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

    run_kwargs = {
        "host_args": [{"cmd": cmd} for cmd in commands],
        "stop_on_errors": False,
    }
    if args.timeout:
        run_kwargs["read_timeout"] = args.timeout
        run_kwargs["channel_timeout"] = args.timeout

    run_kwargs = _filter_kwargs(client.run_command, run_kwargs)
    output = client.run_command("%(cmd)s", **run_kwargs)

    join_kwargs = {"timeout": args.timeout} if args.timeout else {}
    join_kwargs = _filter_kwargs(client.join, join_kwargs)
    client.join(output, **join_kwargs)

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
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", "-1")),
        help="Local rank (torchrun).",
    )
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
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("DIST_TIMEOUT", "40")),
        help="Timeout seconds used for torch.distributed and parallel-ssh.",
    )
    parser.add_argument("--device", default=os.getenv("DIST_DEVICE", "cuda"), help="cpu or cuda.")
    parser.add_argument(
        "--log-dir",
        default=os.getenv("LOG_DIR", "tests/trial/logs"),
        help="Directory for per-rank logs (empty to disable).",
    )
    parser.add_argument("--user", default=os.getenv("SSH_USER"), help="SSH user (optional).")
    parser.add_argument("--identity-file", default=os.getenv("SSH_IDENTITY_FILE"), help="SSH identity file (optional).")
    parser.add_argument("--ssh-port", type=int, default=int(os.getenv("SSH_PORT", "22")), help="SSH port.")
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        help="Extra env vars to pass to workers (repeatable), format KEY=VALUE.",
    )
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
