#!/usr/bin/env bash
set -euo pipefail

tests/scripts/run_intranode.sh "$@"
tests/scripts/run_internode.sh "$@"
tests/scripts/run_low_latency.sh "$@"
