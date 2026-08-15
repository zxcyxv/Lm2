"""Matched H16 control with raw carriers and count-scaled reads."""
from __future__ import annotations

import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_post_update_full_read_h16_attached_stride16_ce_only_13m as h16


h4 = h16.h4
base = h16.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-no-central-rms-sqrtn-"
    "full-read-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
ORIGINAL_PREFLIGHT = base.preflight
H16_FINAL_CHECKS = h16.additional_final_checks


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    transition = model.complex_self_prediction
    if transition.recurrent_hidden_mode != base.RECURRENT_HIDDEN_MODE:
        raise RuntimeError("central recurrence mode differs from registration")
    if not isinstance(transition.input_norm, torch.nn.Identity):
        raise RuntimeError("central QKV RMS normalization is still active")
    if transition.normalize_initial_recurrent_root:
        raise RuntimeError("initial recurrent-root RMS normalization is active")
    if not transition.normalize_accumulated_reads:
        raise RuntimeError("sqrt(write_count) read scaling is disabled")
    if any((
        transition.post_normalize_hidden_reads,
        transition.post_normalize_memory,
        transition.post_normalize_recurrent_state,
    )):
        raise RuntimeError("a central recurrent post-RMS normalization is active")

    generator = torch.Generator().manual_seed(base.SEED + 19191)
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
    diagnostics = {
        "raw_initial_hidden_max_abs_error": float(
            (initial.hidden - roots).abs().max()
        ),
        "raw_initial_memory_max_abs_error": float(
            (initial.memory - transition.write(roots)).detach().abs().max()
        ),
        "raw_successor_memory_max_abs_error": float(
            (first.state.memory - first.raw_updated_memory).abs().max()
        ),
        "raw_successor_hidden_max_abs_error": float(
            (first.state.hidden - first.raw_full_hidden).abs().max()
        ),
        "second_read_sqrtn_formula_max_abs_error": float(
            (second.preliminary_hidden - expected_second_p).abs().max()
        ),
        "parameter_count_no_central_rms": base.parameter_count(model),
    }
    if not all(math.isfinite(float(value)) for value in diagnostics.values()):
        raise RuntimeError("non-finite no-central-RMS diagnostic")
    if any(
        diagnostics[name] != 0.0
        for name in (
            "raw_initial_hidden_max_abs_error",
            "raw_initial_memory_max_abs_error",
            "raw_successor_memory_max_abs_error",
            "raw_successor_hidden_max_abs_error",
            "second_read_sqrtn_formula_max_abs_error",
        )
    ):
        raise RuntimeError("raw-carrier or sqrt(count) formula mismatch")
    if diagnostics["parameter_count_no_central_rms"] != base.EXPECTED_PARAMETERS:
        raise RuntimeError("no-central-RMS configuration changed parameter count")
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
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    del initial_row, rows
    return [
        "## Raw central carriers with count-scaled reads",
        "",
        "Central QKV, initial-root, and recurrent-boundary RMS normalization "
        "were disabled. Memory remained a raw additive carrier; accumulated "
        "memory reads alone used `1/sqrt(write_count)` scaling.",
        "",
        f"Final raw pre-clip gradient norm: {final_row['gradient_norm']:.6f}.",
    ]


for module in (h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.PRE_NORMALIZE_QKV_INPUTS = False
base.NORMALIZE_INITIAL_RECURRENT_ROOT = False
base.NORMALIZE_ACCUMULATED_READS = True
base.POST_NORMALIZE_HIDDEN_READS = False
base.POST_NORMALIZE_MEMORY = False
base.POST_NORMALIZE_RECURRENT_STATE = False
base.EXPECTED_PARAMETERS = 13_215_008
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_no_central_rms_sqrtn_full_read_ce_only_attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The central recurrence used raw QKV inputs, a raw encoder-root copy, "
    "and raw hidden/memory carriers with no central RMS normalization. Only "
    "accumulated-memory reads were divided by sqrt(write_count). The hidden "
    "successor remained the absolute post-update full read."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
