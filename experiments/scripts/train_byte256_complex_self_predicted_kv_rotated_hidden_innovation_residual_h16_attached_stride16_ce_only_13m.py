"""Train H16 with a unitary hidden carrier plus innovation-only delta."""
from __future__ import annotations

import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_no_qkv_prenorm_initial_root_rmsnorm_h16_attached_stride16_ce_only_13m as control


boundary = control.boundary
h16 = control.h16
h4 = control.h4
base = control.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-rotated-hidden-"
    "innovation-residual-boundary-postnorm-no-qkv-prenorm-initial-root-"
    "rmsnorm-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
RECURRENT_HIDDEN_MODE = "rotated-hidden-innovation-residual"
MATCHED_GRADIENT_NORMS = {
    1: 215.86898803710938,
    50: 21.10003089904785,
    100: 68.03108215332031,
    200: 66.59004974365234,
    300: 49.03179931640625,
}
MATCHED_BLOCK_NLLS = {
    1: 6.129311424407206,
    50: 3.3534053801395203,
    100: 3.5582776503633795,
    200: 3.1757790498550094,
    300: 3.163018965928188,
}
POST_WARMUP_GRADIENT_CEILING = 10.0
STEP300_BLOCK_NLL_TOLERANCE = 0.15
ORIGINAL_PREFLIGHT = control.preflight
H16_FINAL_CHECKS = h16.additional_final_checks


def _rms(values: torch.Tensor) -> torch.Tensor:
    return values.float().square().mean(dim=-1).sqrt()


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    transition = model.complex_self_prediction
    if transition.recurrent_hidden_mode != RECURRENT_HIDDEN_MODE:
        raise RuntimeError("unitary hidden residual mode is disabled")
    if transition.residual_prior:
        raise RuntimeError("decoder-facing P unexpectedly has a residual")

    generator = torch.Generator().manual_seed(base.SEED + 18181)
    window = base.sampled_windows(
        training,
        2,
        generator,
        base.TRAIN_WINDOW_LENGTH,
    )
    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        roots = output.prefix_encoded.index_select(1, output.anchor_indices)
        state = transition.initialize(roots)
        step = transition(state)
        expected_raw = step.rotated_hidden + step.innovation_delta
        expected_hidden = transition._fixed_rms_normalize_hidden(expected_raw)
        absolute_full_read = transition.read_to_hidden(
            transition.read(
                transition.queries(step.preliminary_hidden),
                step.raw_updated_memory,
            )
        )

    raw_formula_error = float(
        (step.raw_full_hidden - expected_raw).abs().max()
    )
    boundary_formula_error = float(
        (step.state.hidden - expected_hidden).abs().max()
    )
    full_read_difference = float(
        (step.raw_full_hidden - absolute_full_read).abs().mean()
    )
    raw_rms = _rms(step.raw_full_hidden)
    stored_rms = _rms(step.state.hidden)
    diagnostics = {
        "residual_raw_formula_max_abs_error": raw_formula_error,
        "residual_boundary_formula_max_abs_error": boundary_formula_error,
        "residual_vs_absolute_full_read_mean_abs_difference": (
            full_read_difference
        ),
        "residual_raw_hidden_rms_min": float(raw_rms.min()),
        "residual_raw_hidden_rms_max": float(raw_rms.max()),
        "residual_stored_hidden_rms_min": float(stored_rms.min()),
        "residual_stored_hidden_rms_max": float(stored_rms.max()),
        "innovation_delta_rms_mean": float(
            _rms(step.innovation_delta).mean()
        ),
        "rotated_carrier_rms_mean": float(
            _rms(step.rotated_hidden).mean()
        ),
        "parameter_count_after_hidden_residual": base.parameter_count(model),
    }
    if not all(math.isfinite(float(value)) for value in diagnostics.values()):
        raise RuntimeError("non-finite hidden-residual diagnostic")
    if raw_formula_error != 0.0 or boundary_formula_error != 0.0:
        raise RuntimeError("hidden residual does not implement the registered formula")
    if full_read_difference <= 0.0:
        raise RuntimeError("hidden residual collapsed to the absolute full read")
    if diagnostics["residual_stored_hidden_rms_min"] < 0.999:
        raise RuntimeError("hidden residual failed to anchor the boundary RMS")
    if diagnostics["residual_stored_hidden_rms_max"] > 1.001:
        raise RuntimeError("hidden residual boundary RMS exceeded one")
    if (
        diagnostics["parameter_count_after_hidden_residual"]
        != base.EXPECTED_PARAMETERS
    ):
        raise RuntimeError("hidden residual changed parameter count")
    return {**values, **diagnostics}


def additional_final_checks(initial_row, final_row, rows):
    checks = H16_FINAL_CHECKS(initial_row, final_row, rows)
    rows_by_step = {int(row["step"]): row for row in rows}
    required_steps = (1, 50, 100, 200, 300)
    checks["required_stability_rows_present"] = all(
        step in rows_by_step for step in required_steps
    )
    checks["required_stability_rows_finite"] = all(
        step not in rows_by_step
        or all(
            math.isfinite(float(rows_by_step[step][name]))
            for name in (
                "train_loss",
                "gradient_norm",
                "val_block_validation_nll",
            )
        )
        for step in required_steps
    )
    for step in required_steps:
        if step not in rows_by_step:
            continue
        checks[f"step{step}_gradient_below_matched_control"] = (
            rows_by_step[step]["gradient_norm"]
            < MATCHED_GRADIENT_NORMS[step]
        )
    for step in (50, 100, 200, 300):
        if step not in rows_by_step:
            continue
        checks[f"step{step}_gradient_at_most_10"] = (
            rows_by_step[step]["gradient_norm"]
            <= POST_WARMUP_GRADIENT_CEILING
        )
    if 300 in rows_by_step:
        checks["step300_block_nll_within_matched_tolerance"] = (
            rows_by_step[300]["val_block_validation_nll"]
            <= MATCHED_BLOCK_NLLS[300] + STEP300_BLOCK_NLL_TOLERANCE
        )
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    del initial_row, final_row
    rows_by_step = {int(row["step"]): row for row in rows}
    lines = [
        "## Rotated-hidden innovation residual",
        "",
        "The sole transition change was `Znext = fixed_RMS(RZ + "
        "innovation_delta)`. The CE-facing P, rank-one write, complete "
        "memory update, initial-root normalization, and final Z/S boundary "
        "normalization were retained.",
        "",
        "| Step | Candidate gnorm | Matched gnorm | Candidate block NLL | "
        "Matched block NLL |",
        "|---:|---:|---:|---:|---:|",
    ]
    for step in (1, 50, 100, 200, 300):
        if step not in rows_by_step:
            continue
        row = rows_by_step[step]
        lines.append(
            f"| {step} | {row['gradient_norm']:.6f} | "
            f"{MATCHED_GRADIENT_NORMS[step]:.6f} | "
            f"{row['val_block_validation_nll']:.6f} | "
            f"{MATCHED_BLOCK_NLLS[step]:.6f} |"
        )
    return lines


for module in (control, control.control, boundary, h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.RECURRENT_HIDDEN_MODE = RECURRENT_HIDDEN_MODE
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_rotated_hidden_innovation_residual_boundary_postnorm_"
    "no_qkv_prenorm_initial_root_rmsnorm_ce_only_attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The matched initial-root-RMS H16 recurrence was retained except its "
    "successor hidden is the fixed-RMS-normalized sum of the unitary rotated "
    "previous hidden and the innovation-only read. P, the KV write, complete "
    "memory update, CE, optimizer, and all gradient paths remain unchanged."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
