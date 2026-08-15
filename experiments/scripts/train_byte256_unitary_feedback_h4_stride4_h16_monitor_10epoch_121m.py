"""Scale the H4-supervised state-dependent feedback run to 121M."""
from __future__ import annotations

import math
from pathlib import Path

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch as run


base = run.base

EXPERIMENT_ID = (
    "EXP-20260806-byte256-unitary-feedback-h4-stride4-h16-monitor-"
    "10epoch-width4352-121m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

WIDTH = 4352
EXPECTED_PARAMETERS = 121_336_064
PEAK_LR = 1e-4
WARMUP_STEPS = 500
MASKED_GRADIENT_RELATIVE_TOLERANCE = 6e-3


def scaled_lr_at(index: int, steps: int) -> float:
    """Use the registered lower peak LR and extended warmup."""
    if index < WARMUP_STEPS:
        return PEAK_LR * (index + 1) / WARMUP_STEPS
    progress = (index - WARMUP_STEPS) / max(1, steps - WARMUP_STEPS)
    return PEAK_LR * (0.1 + 0.45 * (1.0 + math.cos(math.pi * progress)))


for module in (run, run.run, run.run.parent, run.run.h16, run.h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.WIDTH = WIDTH
base.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
base.PEAK_LR = PEAK_LR
base.lr_at = scaled_lr_at
run.MASKED_GRADIENT_RELATIVE_TOLERANCE = (
    MASKED_GRADIENT_RELATIVE_TOLERANCE
)
base.REPORT_STEPS = frozenset(
    set(base.REPORT_STEPS) | {10, 25, 500}
)


if __name__ == "__main__":
    base.main()
