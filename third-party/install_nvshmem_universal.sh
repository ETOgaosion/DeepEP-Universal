#!/bin/bash

set -e

cd third-party/nvshmem && \
    mkdir -p build && \
    cmake -DNVSHMEM_PREFIX=/usr/local/nvshmem -DCUDA_HOME=/usr/local/cuda -DNVSHMEM_USE_GDRCOPY=0 -S . -B build && \
    cd build && \
    make -j install


