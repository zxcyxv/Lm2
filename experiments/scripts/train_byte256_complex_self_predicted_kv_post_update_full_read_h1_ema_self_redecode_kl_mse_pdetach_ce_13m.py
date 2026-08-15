"""Train P-detached joint EMA hidden and decoder closure."""
from __future__ import annotations

from pathlib import Path

import train_byte256_complex_self_predicted_kv_post_update_full_read_h1_ema_self_redecode_kl_ce_13m as parent


base = parent.base
EXPERIMENT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "post-update-full-read-h1-ema-self-redecode-kl-mse-pdetach-ce-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

KL_WEIGHT = 1.0
HIDDEN_MSE_WEIGHT = 1.0
KL_ONLY_H1_NLL = 2.347616720944643
KL_ONLY_H1_HIDDEN_MSE = 0.9965372655424289
KL_ONLY_H1_HIDDEN_COSINE = 0.06161094166282055
KL_ONLY_H1_TOP1_AGREEMENT = 0.81787109375
KL_ONLY_H1_KL = 0.7152722680960508


def additional_final_checks(initial_row, final_row, rows):
    del initial_row, rows
    return {
        "h1_hidden_mse_improved_over_kl_only": final_row[
            "val_h1_student_canonical_relative_mse"
        ]
        < KL_ONLY_H1_HIDDEN_MSE,
        "h1_hidden_cosine_improved_over_kl_only": final_row[
            "val_h1_student_canonical_cosine"
        ]
        > KL_ONLY_H1_HIDDEN_COSINE,
        "h1_top1_agreement_preserved_over_kl_only": final_row[
            "val_h1_student_selected_top1_agreement"
        ]
        >= KL_ONLY_H1_TOP1_AGREEMENT,
        "h1_forward_kl_preserved_over_kl_only": final_row[
            "val_h1_self_redecode_forward_kl"
        ]
        <= KL_ONLY_H1_KL,
        "h1_nll_improved_over_kl_only": final_row[
            "val_h1_validation_nll"
        ]
        < KL_ONLY_H1_NLL,
    }


def additional_interpretation_lines(initial_row, final_row, rows):
    del rows
    return [
        "## P-detached joint closure",
        "",
        "The CE-visible P remained attached to ordinary token CE. Both EMA "
        "hidden MSE and EMA decode KL used the same successor graph, whose "
        "K(P), V(P), and Q(P) inputs treated P as a fixed value.",
        "",
        "| Metric | Step 0 | Final | KL-only final |",
        "|---|---:|---:|---:|",
        f"| H1 canonical hidden relative MSE | {initial_row['val_h1_student_canonical_relative_mse']:.6f} | {final_row['val_h1_student_canonical_relative_mse']:.6f} | {KL_ONLY_H1_HIDDEN_MSE:.6f} |",
        f"| H1 canonical hidden cosine | {initial_row['val_h1_student_canonical_cosine']:.6f} | {final_row['val_h1_student_canonical_cosine']:.6f} | {KL_ONLY_H1_HIDDEN_COSINE:.6f} |",
        f"| H1 canonical-to-Znext KL | {initial_row['val_h1_self_redecode_forward_kl']:.6f} | {final_row['val_h1_self_redecode_forward_kl']:.6f} | {KL_ONLY_H1_KL:.6f} |",
        f"| H1 P/Znext top-1 | {initial_row['val_h1_student_selected_top1_agreement']:.6f} | {final_row['val_h1_student_selected_top1_agreement']:.6f} | {KL_ONLY_H1_TOP1_AGREEMENT:.6f} |",
        f"| H1 validation NLL | {initial_row['val_h1_validation_nll']:.6f} | {final_row['val_h1_validation_nll']:.6f} | {KL_ONLY_H1_NLL:.6f} |",
        "",
    ]


parent.KL_WEIGHT = KL_WEIGHT
parent.HIDDEN_MSE_WEIGHT = HIDDEN_MSE_WEIGHT
parent.DETACH_PRELIMINARY_FOR_UPDATE = True

base.EXPERIMENT_ID = EXPERIMENT_ID
base.DEFAULT_RECORD = DEFAULT_RECORD
base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.LATENT_WEIGHT = 1.0
base.CLOSURE_METRIC_NAME = "self_redecode_kl_plus_hidden_mse"
base.CLOSURE_DISPLAY_NAME = "kl+mse"
base.CLOSURE_VALIDATION_METRIC_NAME = (
    "h1_self_redecode_kl_plus_hidden_mse"
)
base.OBJECTIVE_NAME_OVERRIDE = (
    "raw_tied_ce_plus_p_detached_ema_self_redecode_kl_and_hidden_mse"
)
base.CLOSURE_TARGET_NAME_OVERRIDE = (
    "self_selected_token_ema_hidden_and_inverse_decode_distribution"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The self-selected EMA canonical hidden and its immediate inverse-decode "
    "distribution jointly supervised Znext. P stayed attached to ordinary "
    "CE but was detached where it entered successor K/V/Q projections. "
    "Pre-update memory remained attached."
)
base.objective = parent.objective
base.preflight = parent.preflight
base.evaluate = parent.evaluate
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
