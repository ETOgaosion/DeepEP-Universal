# Intra Node NVSHMEM

REQUIREMENTS: read the [guidance](../../guidance/), write your [plans](../plans)
CONTEXT: I cannot successfully run your CUDA p2p codes, please turn to NVSHMEM
    - For NVSHMEM, I have run intra-node bandwidth get test successfully, check [tests/trial/shmem_get_bw.cu](../../../tests/trial/shmem_get_bw.cu)
    - You should use NVSHMEM APIs to implement
TASK:
    - [ ] write a intranode_nvshmem.cu to replace intranode.cu , use nvshmem to finish intra-node all-to-all
    - [ ] integrate and expose its APIs
    - [ ] write a test_intranode_nvshmem.py to test
CONSTRAINTS:
    - You should not consider NVLink, you can assume that we are not intended to support NVLink
    - You should support SM80, SM90 and SM120
    - You should not support any architecture specified features, remove all of them in original codes
