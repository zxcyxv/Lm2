"""Run the unchanged 121M H4-feedback model with peak LR 2e-4."""
from __future__ import annotations

import math
from pathlib import Path

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch_121m as run


base = run.base

EXPERIMENT_ID = (
    "EXP-20260807-byte256-unitary-feedback-h4-width4352-121m-lr2e-4"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

PEAK_LR = 2e-4
WARMUP_STEPS = 500


def scaled_lr_at(index: int, steps: int) -> float:
    """Retain the registered schedule shape with a 2e-4 peak."""
    if index < WARMUP_STEPS:
        return PEAK_LR * (index + 1) / WARMUP_STEPS
    progress = (index - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
    return PEAK_LR * (
        0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))
    )


for module in (
    run,
    run.run,
    run.run.run,
    run.run.run.parent,
    run.run.run.h16,
    run.run.h4,
    base,
):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

run.PEAK_LR = PEAK_LR
base.PEAK_LR = PEAK_LR
base.lr_at = scaled_lr_at


if __name__ == "__main__":
    base.main()
