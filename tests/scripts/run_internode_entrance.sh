#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-1}
WORLD_SIZE=${WORLD_SIZE:-}
NUM_TOKENS=${NUM_TOKENS:-4096}
HIDDEN=${HIDDEN:-7168}
NUM_TOPK_GROUPS=${NUM_TOPK_GROUPS:-}
NUM_TOPK=${NUM_TOPK:-8}
PRESSURE_TEST_MODE=${PRESSURE_TEST_MODE:-0}
NUM_EXPERTS=${NUM_EXPERTS:-256}

if [[ -z "${NUM_GPUS_PER_NODE}" ]]; then
    echo "NUM_GPUS_PER_NODE is required for multinode runs." >&2
    exit 1
fi
args=(
    --num-processes "$NUM_GPUS_PER_NODE"
    --num-tokens "$NUM_TOKENS"
    --hidden "$HIDDEN"
    --num-topk "$NUM_TOPK"
    --pressure-test-mode "$PRESSURE_TEST_MODE"
    --num-experts "$NUM_EXPERTS"
)
if [[ -n "$WORLD_SIZE" ]]; then args+=(--world-size "$WORLD_SIZE"); fi
if [[ -n "$NUM_TOPK_GROUPS" ]]; then args+=(--num-topk-groups "$NUM_TOPK_GROUPS"); fi
"$PYTHON" tests/scripts/run_internode_multinode.py "${args[@]}" "$@"
