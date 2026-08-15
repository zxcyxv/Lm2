"""Ablation 1: restrict the memory write value to the real axis.

Everything else is the registered 13M `unitary-branch-normalized-residual`
H4-loss / H16-monitor recurrence.  Only ``v`` narrows: the memory, its unitary
transport, and the complex read are unchanged, so ``read_width`` and the
readout stay complex.  The LR schedule remains the parent's 33,570-step cosine
so that the first epoch is step-for-step comparable with the preserved parent
metrics; only the executed step count is one epoch.
"""
from __future__ import annotations

from pathlib import Path

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base
h4 = run.h4

EXPERIMENT_ID = (
    "EXP-20260814-byte256-unitary-feedback-h4-real-value-1epoch-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

EPOCH_STEPS = run.EPOCH_STEPS
SCHEDULE_STEPS = run.TOTAL_STEPS
# The complex value projection [2*heads*value_dim, width] becomes
# [heads*value_dim, width]; 8 * 31 * 1344 = 333,312 fewer parameters.
EXPECTED_PARAMETERS = 13_215_008 - 333_312

for module in (run, run.run, run.run.parent, run.run.h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.REAL_VALUE = True
base.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
base.STEPS = EPOCH_STEPS
base.SCHEDULE_STEPS = SCHEDULE_STEPS
base.OBJECTIVE_NAME_OVERRIDE = (
    "four_step_unitary_feedback_branch_rms_frobenius_residual_real_value_"
    "ce_only_attached_stride4_h16_monitor"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "Ablation of the registered branch-normalized feedback recurrence in "
    "which the rank-one write value is real rather than complex. The unitary "
    "memory transport, the Frobenius-normalized complex read, and the H1--H4 "
    "attached CE objective are unchanged."
)


if __name__ == "__main__":
    base.main()
