"""H16 closure of the original unitary SSM as one residual block."""
from __future__ import annotations

import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_input_prenorm_sqrtn_hidden_residual_h16_attached_stride16_ce_only_13m as parent


h16 = parent.h16
h4 = parent.h4
base = parent.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-unitary-block-"
    "residual-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
RECURRENT_HIDDEN_MODE = "unitary-block-residual"
RESIDUAL_STEP_SCALE = 0.1
ORIGINAL_PREFLIGHT = parent.ORIGINAL_PREFLIGHT
H16_FINAL_CHECKS = parent.H16_FINAL_CHECKS


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    transition = model.complex_self_prediction
    if transition.recurrent_hidden_mode != RECURRENT_HIDDEN_MODE:
        raise RuntimeError("unitary block residual mode is disabled")
    if transition.residual_step_scale != RESIDUAL_STEP_SCALE:
        raise RuntimeError("registered residual step scale changed")
    if not transition.pre_normalize_qkv_inputs:
        raise RuntimeError("QKV input pre-normalization is disabled")
    if not transition.normalize_accumulated_reads:
        raise RuntimeError("sqrt(write_count) carrier coordinates are disabled")
    if any((
        transition.normalize_initial_recurrent_root,
        transition.post_normalize_hidden_reads,
        transition.post_normalize_memory,
        transition.post_normalize_recurrent_state,
    )):
        raise RuntimeError("an unregistered state normalization is active")

    generator = torch.Generator().manual_seed(base.SEED + 23232)
    window = base.sampled_windows(
        training, 2, generator, base.TRAIN_WINDOW_LENGTH
    )
    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        roots = output.prefix_encoded.index_select(1, output.anchor_indices)
        initial = transition.initialize(roots)
        first = transition(initial)
        second = transition(first.state)
        first_write = transition.write(first.rotated_hidden)
        first_read = transition.read(
            transition.queries(first.rotated_hidden), first_write
        )
        first_delta = transition.read_to_hidden(first_read)
        expected_first = (
            first.rotated_hidden + RESIDUAL_STEP_SCALE * first_delta
        )
        second_read = transition.read(
            transition.queries(second.rotated_hidden), second.state.memory
        ) / math.sqrt(2.0)
        second_delta = transition.read_to_hidden(second_read)

    diagnostics = {
        "zero_initial_memory_max_abs": float(initial.memory.abs().max()),
        "first_single_write_max_abs_error": float(
            (first.state.memory - first_write).abs().max()
        ),
        "first_block_residual_max_abs_error": float(
            (first.state.hidden - expected_first).abs().max()
        ),
        "decoder_recurrent_state_max_abs_error": float(
            (first.read_hidden - first.state.hidden).abs().max()
        ),
        "second_sqrtn_measurement_max_abs_error": float(
            (second.innovation_delta - second_delta).abs().max()
        ),
        "initial_write_count": initial.write_count,
        "first_write_count": first.state.write_count,
        "second_write_count": second.state.write_count,
        "output_weight_initial_std": float(transition.output.weight.std()),
    }
    for name in (
        "zero_initial_memory_max_abs",
        "first_single_write_max_abs_error",
        "first_block_residual_max_abs_error",
        "decoder_recurrent_state_max_abs_error",
        "second_sqrtn_measurement_max_abs_error",
    ):
        if diagnostics[name] != 0.0:
            raise RuntimeError(f"unitary block formula mismatch: {name}")
    if (initial.write_count, first.state.write_count, second.state.write_count) != (
        0, 1, 2
    ):
        raise RuntimeError("unitary block write count mismatch")
    if not diagnostics["output_weight_initial_std"] > 0.01:
        raise RuntimeError("complex output retained the obsolete tiny init")
    return {**values, **diagnostics}


def additional_final_checks(initial_row, final_row, rows):
    checks = H16_FINAL_CHECKS(initial_row, final_row, rows)
    rows_by_step = {int(row["step"]): row for row in rows}
    for step in (50, 100, 200, 300):
        checks[f"step{step}_gradient_at_most_10"] = (
            step in rows_by_step
            and rows_by_step[step]["gradient_norm"] <= 10.0
        )
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    del initial_row, rows
    return [
        "## Unitary SSM block closure",
        "",
        "Each recurrent step started from the current latent, performed one "
        "write and one updated-memory measurement, and used that measurement "
        "once as a scale-0.1 block-boundary residual. Decoder output and the "
        "next recurrent state were identical.",
        "",
        f"Final raw pre-clip gradient norm: {final_row['gradient_norm']:.6f}.",
    ]


for module in (parent, h16, h4, base):
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
    "sixteen_step_unitary_ssm_single_measurement_block_residual_ce_only_"
    "attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The original unitary SSM was closed as one recurrent residual block: "
    "zero initial memory, one current-latent write, one sqrt(count)-scaled "
    "updated-memory measurement, and one fixed-scale residual shared by the "
    "decoder output and successor state. No detach or post-normalization."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
