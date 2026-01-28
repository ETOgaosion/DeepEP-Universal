FROM nvcr.io/nvidia/pytorch:25.02-py3

RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

RUN wget https://developer.download.nvidia.com/compute/nvshmem/3.5.19/local_installers/nvshmem-local-repo-ubuntu2404-3.5.19_3.5.19-1_amd64.deb && \
    dpkg -i nvshmem-local-repo-ubuntu2404-3.5.19_3.5.19-1_amd64.deb && \
    cp /var/nvshmem-local-repo-ubuntu2404-3.5.19/nvshmem-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y nvshmem-cuda-12

RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir /usr/local/nvshmem && \
    ln -s /usr/bin/nvshmem /usr/local/nvshmem/bin && \
    ln -s /usr/include/nvshmem /usr/local/nvshmem/include && \
    ln -s /usr/lib/x86_64-linux-gnu/nvshmem/12 /usr/local/nvshmem/lib

ENV NVSHMEM_HOME=/usr/local/nvshmem