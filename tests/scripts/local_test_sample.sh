#!/usr/bin/env bash
# Sample local overrides for multinode internode tests.
# Copy to tests/scripts/local_test.sh and customize if needed.

# Required for multinode entrance script
export WORLD_SIZE=2
export NUM_GPUS_PER_NODE=8

# Optional SSH settings for parallel-ssh
# export SSH_USER=your_user
# export SSH_IDENTITY_FILE=~/.ssh/id_rsa
# export PSSH_TIMEOUT=120

# Test parameters (optional overrides)
export NUM_TOKENS=${NUM_TOKENS:-4096}
export HIDDEN=${HIDDEN:-7168}
export NUM_TOPK=${NUM_TOPK:-8}
export NUM_EXPERTS=${NUM_EXPERTS:-256}
export PRESSURE_TEST_MODE=${PRESSURE_TEST_MODE:-0}
# export NUM_TOPK_GROUPS=2
# Optional Python override
# export PYTHON=python
