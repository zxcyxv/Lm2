"""Train the sqrt(n)-normalized complex recurrence with H1 CE only."""
from __future__ import annotations

from pathlib import Path

import train_byte256_complex_self_predicted_kv_post_update_full_read_h4_ce_only_13m as h4


base = h4.base
EXPERIMENT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "post-update-full-read-sqrtn-h1-ce-only-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
H4_STEP200_H1_NLL = 2.168539
MSE_CE_STEP200_H1_NLL = 2.063148


def additional_final_checks(initial_row, final_row, rows):
    del rows
    return {
        "h1_nll_improved": final_row["val_h1_target_nll"]
        < initial_row["val_h1_target_nll"],
        "h1_accuracy_above_random": final_row["val_h1_target_accuracy"]
        > 1.0 / base.VOCABULARY,
        "h1_nll_below_matched_h4_step200": final_row["val_h1_target_nll"]
        < H4_STEP200_H1_NLL,
        "h1_nll_below_mse_ce_step200": final_row["val_h1_target_nll"]
        < MSE_CE_STEP200_H1_NLL,
        "h1_ar_parallel_logit_identity": final_row[
            "val_h1_ar_parallel_max_logit_error"
        ]
        < 5e-4,
    }


def additional_interpretation_lines(initial_row, final_row, rows):
    del rows
    return [
        "## Matched H1 CE-only control",
        "",
        f"H1 validation NLL: {initial_row['val_h1_target_nll']:.6f} -> "
        f"{final_row['val_h1_target_nll']:.6f}.",
        "",
        f"The matched four-horizon CE run reached {H4_STEP200_H1_NLL:.6f} "
        "at step 200; the unnormalized H1 MSE+CE predecessor reached "
        f"{MSE_CE_STEP200_H1_NLL:.6f}.",
        "",
    ]


h4.EXPERIMENT_ID = EXPERIMENT_ID
h4.DEFAULT_RECORD = DEFAULT_RECORD
h4.DEFAULT_OUTPUT = DEFAULT_OUTPUT
h4.TRAIN_HORIZONS = 1
h4.TRAIN_BOUNDARIES = len(
    range(0, base.CONTEXT, base.TRAIN_ANCHOR_STRIDE)
)
base.EXPERIMENT_ID = EXPERIMENT_ID
base.DEFAULT_RECORD = DEFAULT_RECORD
base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.TRAIN_HORIZONS = 1
base.TRAIN_WINDOW_LENGTH = base.CONTEXT + 1
base.TRAIN_ANCHORS = h4.TRAIN_BOUNDARIES
base.REPORT_STEPS = frozenset((1, 50, 100, 200))
base.OBJECTIVE_NAME_OVERRIDE = "h1_central_decode_ce_only_sqrtn_reads"
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The sqrt(n)-normalized post-update full-read recurrence executed one "
    "target-free step. Its innovation-free P state received corpus H1 CE "
    "only; no hidden, KL, EMA, or token-reencoding target was present."
)
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
