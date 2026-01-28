# Plan: Intra-node CUDA P2P check

1. Review intranode dispatch/combine kernels and locate arch-specific (SM90/TMA) paths to remove.
2. Simplify `csrc/kernels/intranode.cu` to keep only the generic CUDA P2P all-to-all path.
3. Rewrite `tests/functional_tests/test_intranode.py` as a minimal P2P all-to-all dispatch + combine validation.
