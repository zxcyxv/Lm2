"""Replicate the registered 13M H4/H16 baseline for 1,000 steps.

Purpose is trajectory variance, not a new result: the preserved parent metrics
were produced on an RTX 5090, while the 2026-08-14 ablations run on an RTX 4000
Ada.  Re-running the unmodified parent on the ablation hardware shows how much
of any ablation gap is hardware/run noise rather than the intervention.

Everything except the executed step count is the parent's: seed, splits,
sampler order, recurrence, loss, batch, and the 33,570-step cosine schedule.

Preflight amendment: the inherited branch-normalized formula audit compares the
packed-GEMM forward memory against a separately executed three-GEMM
recomputation using a fixed 2e-6 *absolute* envelope.  Against values of scale
~4 that is a ~5e-7 relative bound, which is below float32 GEMM reassociation
noise and is therefore hardware dependent; it passed on the RTX 5090 and fails
marginally here.  The measured relative discrepancy is 3.0e-7, i.e. roundoff
rather than a semantic mismatch, so only that one envelope is widened.  The
measured absolute value stays recorded in `run.tsv` as
`first_raw_memory_max_abs_error`, and every other preflight check -- leakage,
masked-H16/truncated-H4 equivalence, gradient finiteness, parameter count --
is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
h4 = run.h4
branch_normalized = run.run

EXPERIMENT_ID = (
    "EXP-20260814-byte256-unitary-feedback-h4-stride4-h16-monitor-"
    "1000step-replication-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

STEPS = 1_000
SCHEDULE_STEPS = run.TOTAL_STEPS
FORMULA_TOLERANCE_BASE = 1e-5

for module in (run, run.run, run.run.parent, run.run.h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

branch_normalized.FORMULA_TOLERANCE_BASE = FORMULA_TOLERANCE_BASE
base.STEPS = STEPS
base.SCHEDULE_STEPS = SCHEDULE_STEPS


if __name__ == "__main__":
    base.main()
