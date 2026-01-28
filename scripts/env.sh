#!/bin/bash
export NVCC_GENCODE="arch=compute_120,code=sm_120"
export NVSHMEM_HOME=/usr/local/nvshmem
export NVSHMEM_DISABLE_GDRCOPY=true
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_DIR=$NVSHMEM_HOME
export PATH="${NVSHMEM_DIR}/bin:$PATH"