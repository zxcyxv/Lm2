"""H16 raw-carrier candidate with a unitary hidden residual."""
from __future__ import annotations

import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_no_central_rms_sqrtn_full_read_h16_attached_stride16_ce_only_13m as control


h16 = control.h16
h4 = control.h4
base = control.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-no-central-rms-sqrtn-"
    "hidden-residual-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
RECURRENT_HIDDEN_MODE = "rotated-hidden-innovation-residual"
ORIGINAL_PREFLIGHT = control.preflight
CONTROL_FINAL_CHECKS = control.additional_final_checks


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    transition = model.complex_self_prediction
    if transition.recurrent_hidden_mode != RECURRENT_HIDDEN_MODE:
        raise RuntimeError("unitary hidden residual is disabled")
    generator = torch.Generator().manual_seed(base.SEED + 20202)
    window = base.sampled_windows(
        training,
        2,
        generator,
        base.TRAIN_WINDOW_LENGTH,
    )
    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        roots = output.prefix_encoded.index_select(1, output.anchor_indices)
        step = transition(transition.initialize(roots))
        expected = step.rotated_hidden + step.innovation_delta
    formula_error = float((step.state.hidden - expected).abs().max())
    if not math.isfinite(formula_error) or formula_error != 0.0:
        raise RuntimeError("raw hidden residual formula mismatch")
    return {
        **values,
        "raw_hidden_residual_formula_max_abs_error": formula_error,
    }


def additional_final_checks(initial_row, final_row, rows):
    checks = CONTROL_FINAL_CHECKS(initial_row, final_row, rows)
    rows_by_step = {int(row["step"]): row for row in rows}
    for step in (50, 100, 200, 300):
        if step in rows_by_step:
            checks[f"step{step}_gradient_at_most_10"] = (
                rows_by_step[step]["gradient_norm"] <= 10.0
            )
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    del initial_row, rows
    return [
        "## Raw unitary hidden residual",
        "",
        "The sole difference from the fresh no-central-RMS sqrt(count) "
        "control was `Znext = RZ + innovation_delta`; no post-normalization "
        "was applied to that sum.",
        "",
        f"Final raw pre-clip gradient norm: {final_row['gradient_norm']:.6f}.",
    ]


for module in (control, h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.RECURRENT_HIDDEN_MODE = RECURRENT_HIDDEN_MODE
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_no_central_rms_sqrtn_unitary_hidden_residual_ce_only_"
    "attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The fresh no-central-RMS sqrt(write_count) control was retained exactly, "
    "except the absolute full-read hidden successor was replaced by the raw "
    "unitary rotated carrier plus innovation-only delta."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
