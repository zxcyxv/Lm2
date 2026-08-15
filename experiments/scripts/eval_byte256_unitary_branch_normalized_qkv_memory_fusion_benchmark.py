"""Run the common fast-path benchmark after QKV and memory fusion."""
from pathlib import Path

import eval_byte256_unitary_branch_normalized_fast_path_benchmark as benchmark


benchmark.RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step6000-qkv-memory-fusion-benchmark"
)


if __name__ == "__main__":
    benchmark.main()
