"""Train attached H4 CE with sparse stride-four training anchors."""
from __future__ import annotations

from pathlib import Path

import train_byte256_complex_self_predicted_kv_post_update_full_read_h4_ce_only_13m as h4


base = h4.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-"
    "post-update-full-read-h4-ce-only-attached-stride4-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
STRIDE1_STEP1000_BLOCK_NLL = 2.245606263727222
STRIDE1_PEAK_VRAM_BYTES = 7_507_823_104


def additional_final_checks(initial_row, final_row, rows):
    del rows
    checks = {
        "block_nll_improved": final_row["val_block_validation_nll"]
        < initial_row["val_block_validation_nll"],
        "block_nll_below_3": final_row["val_block_validation_nll"] < 3.0,
        "peak_vram_below_stride1": final_row["peak_vram_bytes"]
        < STRIDE1_PEAK_VRAM_BYTES,
    }
    for horizon in range(1, h4.TRAIN_HORIZONS + 1):
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
        "## Attached stride-four H4 CE",
        "",
        "Only training anchor stride changed: 64 anchors and 256 CE labels "
        "per sequence replaced 256 anchors and 1024 overlapping labels.",
        "",
        "| Horizon | Initial NLL | Final NLL | Final target accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for horizon in range(1, h4.TRAIN_HORIZONS + 1):
        lines.append(
            f"| H{horizon} | "
            f"{initial_row[f'val_h{horizon}_target_nll']:.6f} | "
            f"{final_row[f'val_h{horizon}_target_nll']:.6f} | "
            f"{final_row[f'val_h{horizon}_target_accuracy']:.6f} |"
        )
    lines.extend(
        (
            "",
            "Stride-four block NLL: "
            f"{initial_row['val_block_validation_nll']:.6f} -> "
            f"{final_row['val_block_validation_nll']:.6f}.",
            f"Stride-one step-1000 block NLL: {STRIDE1_STEP1000_BLOCK_NLL:.6f}.",
            f"Stride-four peak VRAM: {final_row['peak_vram_bytes']:.0f} bytes; "
            f"stride-one peak: {STRIDE1_PEAK_VRAM_BYTES} bytes.",
            "",
        )
    )
    return lines


h4.EXPERIMENT_ID = EXPERIMENT_ID
h4.DEFAULT_RECORD = DEFAULT_RECORD
h4.DEFAULT_OUTPUT = DEFAULT_OUTPUT
h4.DETACH_CROSS_HORIZON_GRADIENTS = False
h4.TRAIN_BOUNDARIES = len(range(0, base.CONTEXT, 4))
base.EXPERIMENT_ID = EXPERIMENT_ID
base.DEFAULT_RECORD = DEFAULT_RECORD
base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.TRAIN_ANCHOR_STRIDE = 4
base.TRAIN_HORIZONS = h4.TRAIN_HORIZONS
base.TRAIN_WINDOW_LENGTH = base.CONTEXT + h4.TRAIN_HORIZONS
base.TRAIN_ANCHORS = h4.TRAIN_BOUNDARIES * h4.TRAIN_HORIZONS
base.OBJECTIVE_NAME_OVERRIDE = (
    "four_step_central_parallel_decode_ce_only_attached_stride4"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The fully attached central recurrence executed four target-free steps "
    "at 64 stride-four prefix anchors. Its four P states received equal token "
    "CE in one causal decode; no hidden, KL, EMA, detach, or token-reencoding "
    "target was present."
)
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
