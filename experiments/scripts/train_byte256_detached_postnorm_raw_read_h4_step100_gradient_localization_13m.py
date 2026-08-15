"""Fresh 100-step precursor for detached-postnorm gradient localization."""
from __future__ import annotations

import math
from pathlib import Path

import train_byte256_unitary_detached_input_raw_read_postnorm_h4_stride4_h16_monitor_10epoch_13m as parent


feedback = parent.parent
base = parent.base
EXPERIMENT_ID = (
    "EXP-20260807-byte256-detached-postnorm-raw-read-h4-"
    "step100-gradient-localization-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
TOTAL_STEPS = 100
SCHEDULE_STEPS = 33_570
REPORT_STEPS = frozenset((1, 50, 100))


def additional_final_checks(initial_row, final_row, rows):
    return {
        "step100_row_present": int(final_row["step"]) == TOTAL_STEPS,
        "all_training_losses_and_gradients_finite": all(
            int(row["step"]) == 0
            or (
                math.isfinite(float(row["train_loss"]))
                and math.isfinite(float(row["gradient_norm"]))
            )
            for row in rows
        ),
    }


def additional_interpretation_lines(initial_row, final_row, rows):
    del initial_row, rows
    return [
        "## Step-100 gradient-localization precursor",
        "",
        "This run preserved the detached-hidden, raw-read, recurrent-postnorm "
        "architecture and the 33,570-step learning-rate schedule, but stopped "
        "after producing the fresh step-100 checkpoint used by the separate "
        "gradient-localization audit.",
        "",
        f"Step-100 raw pre-clip global gradient norm: "
        f"{final_row['gradient_norm']:.6f}.",
    ]


for module in (
    parent,
    feedback,
    parent.branch,
    parent.branch.parent,
    parent.branch.h16,
    parent.branch.h4,
    feedback.h4,
    base,
):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.STEPS = TOTAL_STEPS
base.SCHEDULE_STEPS = SCHEDULE_STEPS
base.REPORT_STEPS = REPORT_STEPS
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
