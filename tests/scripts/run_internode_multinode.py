#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import sys
from pathlib import Path

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


def main(argv=None):
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run internode tests across multiple nodes via parallel-ssh.")
    parser.add_argument("--repo-root", default=str(repo_root), help="Path to DeepEP repo on all nodes.")
    parser.add_argument("--master-port", default=os.getenv("MASTER_PORT", "8361"), help="MASTER_PORT for torch init.")
    parser.add_argument("--user", default=os.getenv("SSH_USER"), help="SSH user (optional).")
    parser.add_argument("--identity-file", default=os.getenv("SSH_IDENTITY_FILE"), help="SSH identity file (optional).")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("PSSH_TIMEOUT", "0")), help="SSH timeout seconds (0 = default).")
    # Note: accept the same arguments as tests/functional_tests/test_internode.py
    parser.add_argument("--num-processes", type=int, default=int(os.getenv("NUM_PROCESSES", "8")))
    parser.add_argument("--world-size", type=int, default=int(os.getenv("WORLD_SIZE", "0")))
    parser.add_argument("--num-tokens", type=int, default=int(os.getenv("NUM_TOKENS", "4096")))
    parser.add_argument("--hidden", type=int, default=int(os.getenv("HIDDEN", "7168")))
    parser.add_argument("--num-topk-groups", type=int, default=int(os.getenv("NUM_TOPK_GROUPS", "0")) or None)
    parser.add_argument("--num-topk", type=int, default=int(os.getenv("NUM_TOPK", "8")))
    parser.add_argument("--pressure-test-mode", type=int, default=int(os.getenv("PRESSURE_TEST_MODE", "0")))
    parser.add_argument("--num-experts", type=int, default=int(os.getenv("NUM_EXPERTS", "256")))
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    cfg = _load_env_config(repo_root)

    master_addr = cfg.get("MASTER_NODE_ADDR")
    slave_addrs = list(cfg.get("SLAVE_NODE_ADDRS", []))
    if not master_addr:
        raise SystemExit("MASTER_NODE_ADDR is missing in .secrets/env.json")
    if not slave_addrs:
        raise SystemExit("SLAVE_NODE_ADDRS is empty in .secrets/env.json")

    hosts = [master_addr] + slave_addrs
    world_size = args.world_size
    if world_size <= 0:
        raise SystemExit("--world-size is required and must be > 0.")
    if world_size != len(hosts):
        raise SystemExit(
            f"--world-size ({world_size}) does not match host list length ({len(hosts)})."
        )

    passthrough_env = {
        "PYTHON": os.getenv("PYTHON"),
        "NUM_PROCESSES": args.num_processes,
        "NUM_TOKENS": args.num_tokens,
        "HIDDEN": args.hidden,
        "NUM_TOPK_GROUPS": args.num_topk_groups,
        "NUM_TOPK": args.num_topk,
        "PRESSURE_TEST_MODE": args.pressure_test_mode,
        "NUM_EXPERTS": args.num_experts,
    }

    def build_cmd(rank: int):
        env_items = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": args.master_port,
            "WORLD_SIZE": world_size,
            "RANK": rank,
            **passthrough_env,
        }
        env_prefix = _build_env_exports(env_items)
        script_args = [
            "--num-processes", str(args.num_processes),
            "--num-tokens", str(args.num_tokens),
            "--hidden", str(args.hidden),
            "--num-topk", str(args.num_topk),
            "--pressure-test-mode", str(args.pressure_test_mode),
            "--num-experts", str(args.num_experts),
        ]
        if args.num_topk_groups is not None:
            script_args += ["--num-topk-groups", str(args.num_topk_groups)]
        python_bin = os.getenv("PYTHON") or "python"
        arg_str = " ".join(shlex.quote(str(a)) for a in script_args)
        conda_sh = os.getenv("CONDA_SH", "$HOME/miniconda3/etc/profile.d/conda.sh")
        conda_env = os.getenv("CONDA_ENV", "deepep")
        return (
            f"cd {shlex.quote(str(repo_root))} && "
            f"source {conda_sh} && conda activate {shlex.quote(conda_env)} && "
            f"{env_prefix} {shlex.quote(python_bin)} "
            f"tests/functional_tests/test_internode.py {arg_str}"
        ).strip()

    client_kwargs = {}
    if args.user:
        client_kwargs["user"] = args.user
    if args.identity_file:
        client_kwargs["pkey"] = args.identity_file
    if args.timeout:
        client_kwargs["timeout"] = args.timeout

    client = ParallelSSHClient(hosts, **client_kwargs)
    commands = [build_cmd(rank) for rank in range(len(hosts))]

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


if __name__ == "__main__":
    raise SystemExit(main())
