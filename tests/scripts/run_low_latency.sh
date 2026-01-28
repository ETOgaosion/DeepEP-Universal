#!/usr/bin/env bash
set -euo pipefail

if [[ -f tests/scripts/local_test.sh ]]; then
    # shellcheck disable=SC1091
    source tests/scripts/local_test.sh
fi

PYTHON=${PYTHON:-python}
NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_TOKENS=${NUM_TOKENS:-128}
HIDDEN=${HIDDEN:-7168}
NUM_TOPK=${NUM_TOPK:-8}
NUM_EXPERTS=${NUM_EXPERTS:-288}
ALLOW_MNNVL=${ALLOW_MNNVL:-}
DISABLE_NVLINK=${DISABLE_NVLINK:-}
USE_LOGFMT=${USE_LOGFMT:-}
PRESSURE_TEST=${PRESSURE_TEST:-}

args=(
    --num-processes "$NUM_PROCESSES"
    --num-tokens "$NUM_TOKENS"
    --hidden "$HIDDEN"
    --num-topk "$NUM_TOPK"
    --num-experts "$NUM_EXPERTS"
)
if [[ -n "$ALLOW_MNNVL" ]]; then args+=(--allow-mnnvl); fi
if [[ -n "$DISABLE_NVLINK" ]]; then args+=(--disable-nvlink); fi
if [[ -n "$USE_LOGFMT" ]]; then args+=(--use-logfmt); fi
if [[ -n "$PRESSURE_TEST" ]]; then args+=(--pressure-test); fi

"$PYTHON" tests/functional_tests/test_low_latency.py "${args[@]}" "$@"
