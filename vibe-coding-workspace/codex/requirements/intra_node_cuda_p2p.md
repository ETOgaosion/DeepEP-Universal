# Intra Node CUDA p2p check

REQUIREMENTS: read the [guidance](../../guidance/), write your [plans](../plans)
CONTEXT: Original intra node test is too hard to run successfully, you should rewrite
TASK:
    - [ ] refer to intranode cuda implementation, disable all unnecessary features, only remain original all-to-all function
    - [ ] You should change [csrc/kernels/intranode.cu](../../../csrc/kernels/intranode.cu) to only use CUDA p2p
    - [ ] write your test_intranode.py
CONSTRAINTS:
    - You should not consider NVLink, you can assume that we are not intended to support NVLink
    - You should use CUDA P2P API in [csrc/kernels/intranode.cu](../../../csrc/kernels/intranode.cu), You can assume that we have hacked 5090 driver to support CUDA P2P through PCIe
    - You should support SM80, SM90 and SM120
    - You should not support any architecture specified features, remove all of them in original codes
