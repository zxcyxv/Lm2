"""Ablate the compiled memory region while retaining packed single-GEMM QKV."""
from pathlib import Path

import eval_byte256_unitary_branch_normalized_fast_path_benchmark as benchmark


benchmark.base.FUSE_BRANCH_MEMORY_KERNEL = False
benchmark.RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step6000-qkv-memory-fusion-benchmark/qkv_only"
)


if __name__ == "__main__":
    benchmark.main()
