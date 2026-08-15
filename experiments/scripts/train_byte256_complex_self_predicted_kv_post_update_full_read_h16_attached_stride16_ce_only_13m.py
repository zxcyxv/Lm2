"""Train attached H16 CE with non-overlapping stride-sixteen anchors."""
from __future__ import annotations

from pathlib import Path

import train_byte256_complex_self_predicted_kv_post_update_full_read_h4_ce_only_13m as h4


base = h4.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-"
    "post-update-full-read-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
TRAIN_HORIZONS = 16
ANCHOR_STRIDE = 16
H4_STRIDE4_BLOCK_NLL = 2.385889
H4_STRIDE4_PEAK_VRAM_BYTES = 2_816_060_416


def additional_final_checks(initial_row, final_row, rows):
    del rows
    checks = {
        "block_nll_improved": final_row["val_block_validation_nll"]
        < initial_row["val_block_validation_nll"],
        "block_nll_below_4": final_row["val_block_validation_nll"] < 4.0,
    }
    for horizon in range(1, TRAIN_HORIZONS + 1):
        checks[f"h{horizon}_nll_improved"] = final_row[
            f"val_h{horizon}_target_nll"
        ] < initial_row[f"val_h{horizon}_target_nll"]
        checks[f"h{horizon}_accuracy_above_random"] = final_row[
            f"val_h{horizon}_target_accuracy"
        ] > 1.0 / base.VOCABULARY
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    del rows
    lines = [
        "## Attached stride-sixteen H16 CE",
        "",
        "Sixteen stride-16 anchors and sixteen horizons retained 256 CE "
        "labels per sequence while increasing central sequential depth.",
        "",
        "| Horizon | Initial NLL | Final NLL | Final target accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for horizon in range(1, TRAIN_HORIZONS + 1):
        lines.append(
            f"| H{horizon} | "
            f"{initial_row[f'val_h{horizon}_target_nll']:.6f} | "
            f"{final_row[f'val_h{horizon}_target_nll']:.6f} | "
            f"{final_row[f'val_h{horizon}_target_accuracy']:.6f} |"
        )
    lines.extend(
        (
            "",
            "H16 block NLL: "
            f"{initial_row['val_block_validation_nll']:.6f} -> "
            f"{final_row['val_block_validation_nll']:.6f}.",
            f"H4/stride-4 block NLL: {H4_STRIDE4_BLOCK_NLL:.6f}.",
            f"H16 peak VRAM: {final_row['peak_vram_bytes']:.0f} bytes; "
            f"H4/stride-4 peak: {H4_STRIDE4_PEAK_VRAM_BYTES} bytes.",
            "",
        )
    )
    return lines


h4.EXPERIMENT_ID = EXPERIMENT_ID
h4.DEFAULT_RECORD = DEFAULT_RECORD
h4.DEFAULT_OUTPUT = DEFAULT_OUTPUT
h4.TRAIN_HORIZONS = TRAIN_HORIZONS
h4.TRAIN_BOUNDARIES = len(range(0, base.CONTEXT, ANCHOR_STRIDE))
h4.DETACH_CROSS_HORIZON_GRADIENTS = False
base.EXPERIMENT_ID = EXPERIMENT_ID
base.DEFAULT_RECORD = DEFAULT_RECORD
base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.HORIZONS = TRAIN_HORIZONS
base.TRAIN_HORIZONS = TRAIN_HORIZONS
base.TRAIN_ANCHOR_STRIDE = ANCHOR_STRIDE
base.EVAL_ANCHOR_STRIDE = ANCHOR_STRIDE
base.TRAIN_WINDOW_LENGTH = base.CONTEXT + TRAIN_HORIZONS
base.EVAL_WINDOW_LENGTH = base.CONTEXT + TRAIN_HORIZONS
base.TRAIN_ANCHORS = h4.TRAIN_BOUNDARIES * TRAIN_HORIZONS
base.METRIC_NAMES = (
    *base.metric_names_for_horizons(TRAIN_HORIZONS),
    "no_auxiliary_loss",
)
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_central_parallel_decode_ce_only_attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The fully attached central recurrence executed sixteen target-free "
    "steps at sixteen non-overlapping stride-16 prefix anchors. Its sixteen "
    "P states received equal token CE in one causal decode; no hidden, KL, "
    "EMA, detach, or token-reencoding target was present."
)
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
