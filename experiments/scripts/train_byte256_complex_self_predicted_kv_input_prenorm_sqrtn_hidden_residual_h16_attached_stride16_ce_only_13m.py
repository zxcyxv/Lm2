"""H16 input-pre-norm, sqrt(count), raw hidden-residual candidate."""
from __future__ import annotations

import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_post_update_full_read_h16_attached_stride16_ce_only_13m as h16


h4 = h16.h4
base = h16.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-input-prenorm-sqrtn-"
    "hidden-residual-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
RECURRENT_HIDDEN_MODE = "rotated-hidden-innovation-residual"
ORIGINAL_PREFLIGHT = base.preflight
H16_FINAL_CHECKS = h16.additional_final_checks


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    transition = model.complex_self_prediction
    if transition.recurrent_hidden_mode != RECURRENT_HIDDEN_MODE:
        raise RuntimeError("unitary hidden residual is disabled")
    if not transition.pre_normalize_qkv_inputs:
        raise RuntimeError("QKV input pre-normalization is disabled")
    if not isinstance(transition.input_norm, torch.nn.RMSNorm):
        raise RuntimeError("QKV input pre-normalization is not RMSNorm")
    if transition.normalize_initial_recurrent_root:
        raise RuntimeError("private initial-root normalization is active")
    if not transition.normalize_accumulated_reads:
        raise RuntimeError("sqrt(write_count) read scaling is disabled")
    if any((
        transition.post_normalize_hidden_reads,
        transition.post_normalize_memory,
        transition.post_normalize_recurrent_state,
    )):
        raise RuntimeError("a recurrent post-normalization is active")

    generator = torch.Generator().manual_seed(base.SEED + 21212)
    window = base.sampled_windows(
        training,
        2,
        generator,
        base.TRAIN_WINDOW_LENGTH,
    )
    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        roots = output.prefix_encoded.index_select(1, output.anchor_indices)
        initial = transition.initialize(roots)
        first = transition(initial)
        second = transition(first.state)
        expected_second_p = transition.read_to_hidden(
            transition.read(
                transition.queries(second.rotated_hidden),
                second.rotated_memory,
            )
            / math.sqrt(first.state.write_count)
        )
        expected_first_z = first.rotated_hidden + first.innovation_delta

    diagnostics = {
        "raw_initial_hidden_max_abs_error": float(
            (initial.hidden - roots).abs().max()
        ),
        "raw_initial_memory_max_abs_error": float(
            (initial.memory - transition.write(roots)).detach().abs().max()
        ),
        "raw_hidden_residual_formula_max_abs_error": float(
            (first.state.hidden - expected_first_z).abs().max()
        ),
        "raw_successor_memory_max_abs_error": float(
            (first.state.memory - first.raw_updated_memory).abs().max()
        ),
        "second_read_sqrtn_formula_max_abs_error": float(
            (second.preliminary_hidden - expected_second_p).abs().max()
        ),
        "parameter_count_input_prenorm_residual": base.parameter_count(model),
    }
    if not all(math.isfinite(float(value)) for value in diagnostics.values()):
        raise RuntimeError("non-finite input-pre-norm residual diagnostic")
    if any(
        diagnostics[name] != 0.0
        for name in (
            "raw_initial_hidden_max_abs_error",
            "raw_initial_memory_max_abs_error",
            "raw_hidden_residual_formula_max_abs_error",
            "raw_successor_memory_max_abs_error",
            "second_read_sqrtn_formula_max_abs_error",
        )
    ):
        raise RuntimeError("registered raw residual or sqrt(count) formula mismatch")
    if (
        diagnostics["parameter_count_input_prenorm_residual"]
        != base.EXPECTED_PARAMETERS
    ):
        raise RuntimeError("input-pre-norm residual changed parameter count")
    return {**values, **diagnostics}


def additional_final_checks(initial_row, final_row, rows):
    checks = H16_FINAL_CHECKS(initial_row, final_row, rows)
    rows_by_step = {int(row["step"]): row for row in rows}
    required = (1, 50, 100, 200, 300)
    checks["required_stability_rows_present"] = all(
        step in rows_by_step for step in required
    )
    checks["required_stability_rows_finite"] = all(
        step not in rows_by_step
        or all(
            math.isfinite(float(rows_by_step[step][field]))
            for field in (
                "train_loss",
                "gradient_norm",
                "val_block_validation_nll",
            )
        )
        for step in required
    )
    for step in (50, 100, 200, 300):
        if step in rows_by_step:
            checks[f"step{step}_gradient_at_most_10"] = (
                rows_by_step[step]["gradient_norm"] <= 10.0
            )
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    del initial_row, rows
    return [
        "## Input-pre-norm raw hidden residual",
        "",
        "Only real hidden inputs to the shared Q/K/V projections were "
        "RMS-normalized. Memory remained raw and accumulated-memory reads "
        "used `1/sqrt(write_count)`. The raw successor was `RZ + "
        "innovation_delta` with no recurrent post-normalization.",
        "",
        f"Final raw pre-clip gradient norm: {final_row['gradient_norm']:.6f}.",
    ]


for module in (h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.RECURRENT_HIDDEN_MODE = RECURRENT_HIDDEN_MODE
base.PRE_NORMALIZE_QKV_INPUTS = True
base.NORMALIZE_INITIAL_RECURRENT_ROOT = False
base.NORMALIZE_ACCUMULATED_READS = True
base.POST_NORMALIZE_HIDDEN_READS = False
base.POST_NORMALIZE_MEMORY = False
base.POST_NORMALIZE_RECURRENT_STATE = False
base.EXPECTED_PARAMETERS = 13_216_352
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_input_prenorm_sqrtn_unitary_hidden_residual_ce_only_"
    "attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "Only hidden inputs to central Q/K/V were pre-RMS-normalized. Raw memory "
    "used sqrt(write_count)-scaled accumulated reads, and the raw hidden "
    "successor was the unitary rotated carrier plus innovation-only delta. "
    "No initial-root or recurrent post-normalization was present."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
