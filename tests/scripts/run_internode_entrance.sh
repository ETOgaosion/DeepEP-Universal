#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-1}
WORLD_SIZE=${WORLD_SIZE:-2}
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
if [[ -z "${WORLD_SIZE}" ]]; then
    echo "WORLD_SIZE is required for multinode runs." >&2
    exit 1
fi

args=(
    --num-processes "$NUM_GPUS_PER_NODE"
    --world-size "$WORLD_SIZE"
    --num-tokens "$NUM_TOKENS"
    --hidden "$HIDDEN"
    --num-topk "$NUM_TOPK"
    --pressure-test-mode "$PRESSURE_TEST_MODE"
    --num-experts "$NUM_EXPERTS"
)
if [[ -n "$NUM_TOPK_GROUPS" ]]; then args+=(--num-topk-groups "$NUM_TOPK_GROUPS"); fi
"$PYTHON" tests/scripts/run_internode_multinode.py "${args[@]}" "$@"
