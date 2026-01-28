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
NUM_TOPK=${NUM_TOPK:-8}
NUM_EXPERTS=${NUM_EXPERTS:-256}

args=(
    --num-processes "$NUM_PROCESSES"
    --num-tokens "$NUM_TOKENS"
    --hidden "$HIDDEN"
    --num-topk "$NUM_TOPK"
    --num-experts "$NUM_EXPERTS"
)

"$PYTHON" tests/functional_tests/test_intranode.py "${args[@]}" "$@"
