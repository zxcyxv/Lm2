"""Run the preserved real-sample audit producer on the step-6000 checkpoint."""
from pathlib import Path

import eval_byte256_unitary_branch_normalized_step1000_interpretability as audit


audit.CHECKPOINT = Path(
    "outputs/experiments/EXP-20260802-byte256-unitary-branch-normalized-"
    "h16-6000step-micro64/step6000.pt"
)
audit.RECORD = Path(
    "experiments/records/EXP-20260802-byte256-unitary-branch-normalized-"
    "step6000-interpretability-audit"
)


if __name__ == "__main__":
    audit.main()

