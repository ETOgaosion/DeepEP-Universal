#!/bin/bash

set -e

wget https://developer.download.nvidia.com/compute/nvshmem/3.5.19/local_installers/nvshmem-local-repo-ubuntu2404-3.5.19_3.5.19-1_amd64.deb && \
    dpkg -i nvshmem-local-repo-ubuntu2404-3.5.19_3.5.19-1_amd64.deb && \
    sudo cp /var/nvshmem-local-repo-ubuntu2404-3.5.19/nvshmem-*-keyring.gpg /usr/share/keyrings/ && \
    sudo apt-get update && \
    sudo apt-get install -y nvshmem-cuda-12

sudo mkdir /usr/local/nvshmem && \
    sudo ln -s /usr/bin/nvshmem /usr/local/nvshmem/bin && \
    sudo ln -s /usr/include/nvshmem /usr/local/nvshmem/include && \
    sudo ln -s /usr/lib/x86_64-linux-gnu/nvshmem/12 /usr/local/nvshmem/lib