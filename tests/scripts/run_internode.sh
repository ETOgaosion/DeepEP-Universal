#!/usr/bin/env bash
set -euo pipefail

if [[ -f tests/scripts/local_test.sh ]]; then
    # shellcheck disable=SC1091
    source tests/scripts/local_test.sh
fi

PYTHON=${PYTHON:-python}
NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_TOKENS=${NUM_TOKENS:-4096}
HIDDEN=${HIDDEN:-7168}
NUM_TOPK_GROUPS=${NUM_TOPK_GROUPS:-}
NUM_TOPK=${NUM_TOPK:-8}
PRESSURE_TEST_MODE=${PRESSURE_TEST_MODE:-0}
NUM_EXPERTS=${NUM_EXPERTS:-256}
TEST_LL_COMPATIBILITY=${TEST_LL_COMPATIBILITY:-}

args=(
    --num-processes "$NUM_PROCESSES"
    --num-tokens "$NUM_TOKENS"
    --hidden "$HIDDEN"
    --num-topk "$NUM_TOPK"
    --pressure-test-mode "$PRESSURE_TEST_MODE"
    --num-experts "$NUM_EXPERTS"
)
if [[ -n "$NUM_TOPK_GROUPS" ]]; then args+=(--num-topk-groups "$NUM_TOPK_GROUPS"); fi
if [[ -n "$TEST_LL_COMPATIBILITY" ]]; then args+=(--test-ll-compatibility); fi

"$PYTHON" tests/functional_tests/test_internode.py "${args[@]}" "$@"
