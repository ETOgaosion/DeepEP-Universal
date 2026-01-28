# Install NVSHMEM

Universal DeepEP has no need to rely on high-end network config, like IBGDA/GDRCopy/NVLink , so commonly PCIe machines are supported (performance is minor).

We directly use official NVSHMEM 3.5.19-1 , refer to 

```sh
wget https://developer.download.nvidia.com/compute/nvshmem/3.5.19/local_installers/nvshmem-local-repo-ubuntu2404-3.5.19_3.5.19-1_amd64.deb && \
    dpkg -i nvshmem-local-repo-ubuntu2404-3.5.19_3.5.19-1_amd64.deb && \
    cp /var/nvshmem-local-repo-ubuntu2404-3.5.19/nvshmem-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y nvshmem-cuda-12
```

Create an NVSHMEM_HOME directory and add links of bin/include/lib for other program use:

```sh
mkdir /usr/local/nvshmem && \
    ln -s /usr/bin/nvshmem /usr/local/nvshmem/bin && \
    ln -s /usr/include/nvshmem /usr/local/nvshmem/include && \
    ln -s /usr/lib/x86_64-linux-gnu/nvshmem/12 /usr/local/nvshmem/lib
```

So, your NVSHMEM_HOME=/usr/local/nvshmem