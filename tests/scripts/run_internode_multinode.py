#!/usr/bin/env python3
import argparse
import inspect
import json
import logging
import os
import shlex
import sys
import time
import signal
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

def _collect_extra_env(cfg, args):
    extra_env = {}
    if isinstance(cfg.get("EXTRA_ENVS"), dict):
        extra_env.update(cfg.get("EXTRA_ENVS"))
    test_envs = cfg.get("TEST_ENVS", {})
    if isinstance(test_envs, dict):
        for key in ("run_internode_multinode", "internode_multinode", "internode"):
            if isinstance(test_envs.get(key), dict):
                extra_env.update(test_envs.get(key))
    extra_env.update(_parse_kv(args.extra_env))
    return extra_env

def _resolve_hosts(cfg):
    ssh_master_addr = cfg.get("SSH_MASTER_NODE_ADDR") or cfg.get("MASTER_NODE_IP")
    ssh_slave_addrs = list(cfg.get("SSH_SLAVE_NODE_ADDRS", [])) or list(cfg.get("SLAVE_NODE_ADDRS", []))
    master_addr = cfg.get("MASTER_NODE_IP")
    if not ssh_master_addr:
        raise SystemExit("SSH_MASTER_NODE_ADDR (or MASTER_NODE_IP) is missing in .secrets/env.json")
    if not ssh_slave_addrs:
        raise SystemExit("SSH_SLAVE_NODE_ADDRS (or SLAVE_NODE_ADDRS) is empty in .secrets/env.json")
    if not master_addr:
        raise SystemExit("MASTER_NODE_IP is missing in .secrets/env.json")
    return [ssh_master_addr] + ssh_slave_addrs, master_addr

def main(argv=None):
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run internode tests across multiple nodes via parallel-ssh.")
    parser.add_argument("--repo-root", default=str(repo_root), help="Path to DeepEP repo on all nodes.")
    parser.add_argument("--master-port", default=os.getenv("MASTER_PORT", "8361"), help="MASTER_PORT for torch init.")
    parser.add_argument("--user", default=os.getenv("SSH_USER"), help="SSH user (optional).")
    parser.add_argument("--identity-file", default=os.getenv("SSH_IDENTITY_FILE"), help="SSH identity file (optional).")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("PSSH_TIMEOUT", "0")), help="SSH timeout seconds (0 = default).")
    parser.add_argument("--read-timeout", type=int, default=int(os.getenv("PSSH_READ_TIMEOUT", "0")), help="SSH read timeout seconds (0 = default).")
    parser.add_argument("--channel-timeout", type=int, default=int(os.getenv("PSSH_CHANNEL_TIMEOUT", "0")), help="SSH channel timeout seconds (0 = default).")
    parser.add_argument("--join-timeout", type=int, default=int(os.getenv("PSSH_JOIN_TIMEOUT", "0")), help="Join timeout seconds (0 = default).")
    parser.add_argument("--overall-timeout", type=int, default=int(os.getenv("OVERALL_TIMEOUT", "0")), help="Overall timeout seconds (0 = unlimited).")
    parser.add_argument("--ssh-port", type=int, default=int(os.getenv("SSH_PORT", "22")), help="SSH port.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and verbose diagnostics.")
    parser.add_argument("--preflight-cmd", default=os.getenv("PSSH_PREFLIGHT_CMD", ""), help="Optional command to run on all hosts before the test.")
    # Note: accept the same arguments as tests/functional_tests/test_internode.py
    parser.add_argument("--num-processes", type=int, default=int(os.getenv("NUM_PROCESSES", "8")))
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.getenv("WORLD_SIZE", "0")),
        help="Number of nodes (defaults to host list length).",
    )
    parser.add_argument("--num-tokens", type=int, default=int(os.getenv("NUM_TOKENS", "4096")))
    parser.add_argument("--hidden", type=int, default=int(os.getenv("HIDDEN", "7168")))
    parser.add_argument("--num-topk-groups", type=int, default=int(os.getenv("NUM_TOPK_GROUPS", "0")) or None)
    parser.add_argument("--num-topk", type=int, default=int(os.getenv("NUM_TOPK", "8")))
    parser.add_argument("--pressure-test-mode", type=int, default=int(os.getenv("PRESSURE_TEST_MODE", "0")))
    parser.add_argument("--num-experts", type=int, default=int(os.getenv("NUM_EXPERTS", "256")))
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        help="Extra env vars to pass to workers (repeatable), format KEY=VALUE.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    cfg = _load_env_config(repo_root)

    hosts, master_addr = _resolve_hosts(cfg)
    num_nodes = args.world_size if args.world_size > 0 else len(hosts)
    if args.world_size > 0 and num_nodes != len(hosts):
        raise SystemExit(
            f"--world-size ({num_nodes}) does not match host list length ({len(hosts)})."
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
    extra_env = _collect_extra_env(cfg, args)

    def build_cmd(rank: int):
        env_items = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": args.master_port,
            "WORLD_SIZE": num_nodes,
            "RANK": rank,
            "NNODES": num_nodes,
            "NPROC_PER_NODE": args.num_processes,
            "PYTHONUNBUFFERED": "1",
            **passthrough_env,
            **extra_env,
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
        env_sh = os.getenv("ENV_SH", "scripts/env.sh")
        return (
            f"cd {shlex.quote(str(repo_root))} && "
            f"source {conda_sh} && conda activate {shlex.quote(conda_env)} && "
            f"source {shlex.quote(env_sh)} && "
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
    if args.ssh_port:
        client_kwargs["port"] = args.ssh_port

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        print("[debug] hosts:", hosts)
        print("[debug] master_addr:", master_addr)
        print("[debug] num_nodes:", num_nodes)
        print("[debug] repo_root:", repo_root)
        print("[debug] pssh kwargs:", client_kwargs)

    client = ParallelSSHClient(hosts, **client_kwargs)
    commands = [build_cmd(rank) for rank in range(len(hosts))]

    for host, cmd in zip(hosts, commands):
        print(f"[{host}] cmd: {cmd}")

    run_kwargs = {
        "host_args": [{"cmd": cmd} for cmd in commands],
        "stop_on_errors": False,
    }
    if args.timeout:
        run_kwargs["timeout"] = args.timeout
    if args.read_timeout:
        run_kwargs["read_timeout"] = args.read_timeout
    if args.channel_timeout:
        run_kwargs["channel_timeout"] = args.channel_timeout

    if args.preflight_cmd:
        preflight_kwargs = dict(run_kwargs)
        preflight_kwargs["host_args"] = [{"cmd": args.preflight_cmd} for _ in commands]
        preflight_kwargs = _filter_kwargs(client.run_command, preflight_kwargs)
        preflight_output = client.run_command("%(cmd)s", **preflight_kwargs)
        join_kwargs = {"timeout": args.join_timeout} if args.join_timeout else {}
        join_kwargs = _filter_kwargs(client.join, join_kwargs)
        client.join(preflight_output, **join_kwargs)
        for host, host_output in zip(hosts, preflight_output):
            for line in (host_output.stdout or []):
                print(f"[{host}][preflight] {line}")
            for line in (host_output.stderr or []):
                print(f"[{host}][preflight][stderr] {line}", file=sys.stderr)

    run_kwargs = _filter_kwargs(client.run_command, run_kwargs)
    output = client.run_command("%(cmd)s", **run_kwargs)

    join_kwargs = {"timeout": args.join_timeout} if args.join_timeout else {}
    join_kwargs = _filter_kwargs(client.join, join_kwargs)

    def _timeout_handler(_signum, _frame):
        raise TimeoutError

    start_time = time.monotonic()
    if args.overall_timeout:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.overall_timeout)
    try:
        client.join(output, **join_kwargs)
    except TimeoutError:
        elapsed = int(time.monotonic() - start_time)
        print(f"[timeout] overall timeout exceeded after {elapsed}s", file=sys.stderr)
        return 2
    finally:
        if args.overall_timeout:
            signal.alarm(0)

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
