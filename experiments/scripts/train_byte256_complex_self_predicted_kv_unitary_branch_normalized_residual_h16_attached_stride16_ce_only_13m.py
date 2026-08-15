"""H16 unitary carriers with normalized coupling branches and raw residual."""
from __future__ import annotations

from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_input_prenorm_sqrtn_hidden_residual_h16_attached_stride16_ce_only_13m as parent


h16 = parent.h16
h4 = parent.h4
base = parent.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-unitary-branch-"
    "normalized-residual-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
RECURRENT_HIDDEN_MODE = "unitary-branch-normalized-residual"
# Absolute float32 roundoff envelope for the two-GEMM formula audit below.
# It is a module constant so a replication on different hardware can widen it
# explicitly, the way the 121M run widened it for a wider reduction.
FORMULA_TOLERANCE_BASE = 2e-6
ORIGINAL_PREFLIGHT = parent.ORIGINAL_PREFLIGHT
H16_FINAL_CHECKS = parent.H16_FINAL_CHECKS


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model, training, latent_target_model=latent_target_model
    )
    transition = model.complex_self_prediction
    if transition.recurrent_hidden_mode != RECURRENT_HIDDEN_MODE:
        raise RuntimeError("branch-normalized unitary mode is disabled")
    if not transition.pre_normalize_qkv_inputs:
        raise RuntimeError("fixed QKV branch RMS is disabled")
    if not isinstance(transition.input_norm, torch.nn.Identity):
        raise RuntimeError("QKV branch RMS unexpectedly has learned affine")
    if transition.normalize_accumulated_reads:
        raise RuntimeError("count-based read scaling is active")
    if any((
        transition.normalize_initial_recurrent_root,
        transition.post_normalize_hidden_reads,
        transition.post_normalize_memory,
        transition.post_normalize_recurrent_state,
    )):
        raise RuntimeError("a carrier normalization is active")
    if transition.residual_step_scale != 0.1:
        raise RuntimeError("unused legacy residual scale changed")

    generator = torch.Generator().manual_seed(base.SEED + 24242)
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
        first_frob = first_write.abs().square().sum(
            dim=(-3, -2, -1)
        ).sqrt()
        first_measurement = transition.read(
            transition.queries(first.rotated_hidden), first_write
        ) / (first_frob[..., None, None] + transition.post_norm_eps)
        first_delta = transition.read_to_hidden(first_measurement)
        first_raw = first.rotated_hidden + first_delta
        first_decode = transition._fixed_rms_normalize_hidden(first_raw)

        second_frob = second.state.memory.abs().square().sum(
            dim=(-3, -2, -1)
        ).sqrt()
        second_measurement = transition.read(
            transition.queries(second.rotated_hidden), second.state.memory
        ) / (second_frob[..., None, None] + transition.post_norm_eps)
        second_delta = transition.read_to_hidden(second_measurement)

    diagnostics = {
        "zero_initial_memory_max_abs": float(initial.memory.abs().max()),
        "first_raw_memory_max_abs_error": float(
            (first.state.memory - first_write).abs().max()
        ),
        "first_frobenius_measurement_max_abs_error": float(
            (first.innovation_delta - first_delta).abs().max()
        ),
        "first_ungated_residual_max_abs_error": float(
            (first.state.hidden - first_raw).abs().max()
        ),
        "first_decode_only_rms_max_abs_error": float(
            (first.read_hidden - first_decode).abs().max()
        ),
        "second_frobenius_measurement_max_abs_error": float(
            (second.innovation_delta - second_delta).abs().max()
        ),
        "initial_write_count": initial.write_count,
        "first_write_count": first.state.write_count,
        "second_write_count": second.state.write_count,
        "output_weight_initial_std": float(
            transition.output.weight.detach().std()
        ),
        "parameter_count_branch_normalized": base.parameter_count(model),
    }
    # The memory write is recomputed through a separate float32 GEMM for this
    # formula audit.  Its absolute roundoff envelope grows with the reduction
    # width; preserve the original 13M threshold and scale only that envelope.
    formula_tolerance = FORMULA_TOLERANCE_BASE * max(
        1.0, transition.width / 1344
    )
    for name in (
        "zero_initial_memory_max_abs",
        "first_raw_memory_max_abs_error",
        "first_frobenius_measurement_max_abs_error",
        "first_ungated_residual_max_abs_error",
        "first_decode_only_rms_max_abs_error",
        "second_frobenius_measurement_max_abs_error",
    ):
        if diagnostics[name] > formula_tolerance:
            raise RuntimeError(f"branch-normalized formula mismatch: {name}")
    if (initial.write_count, first.state.write_count, second.state.write_count) != (
        0, 1, 2
    ):
        raise RuntimeError("branch-normalized write count mismatch")
    if diagnostics["output_weight_initial_std"] <= 0.01:
        raise RuntimeError("output retained tiny initialization")
    if diagnostics["parameter_count_branch_normalized"] != base.EXPECTED_PARAMETERS:
        raise RuntimeError("unexpected branch-normalized parameter count")
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
        "## Branch-normalized unitary recurrence",
        "",
        "Raw latent and memory carriers used unitary transport. Fixed RMS was "
        "applied only before Q/K/V, the measurement was divided by the actual "
        "memory Frobenius norm, and the ungated measurement correction was "
        "added to the raw latent carrier. Only the decoder branch RMS-normalized "
        "the successor.",
        "",
        f"Final raw pre-clip gradient norm: {final_row['gradient_norm']:.6f}.",
    ]


for module in (parent, h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.RECURRENT_HIDDEN_MODE = RECURRENT_HIDDEN_MODE
base.FUSE_BRANCH_MEMORY_KERNEL = False
base.PRE_NORMALIZE_QKV_INPUTS = True
base.NORMALIZE_INITIAL_RECURRENT_ROOT = False
base.NORMALIZE_ACCUMULATED_READS = False
base.POST_NORMALIZE_HIDDEN_READS = False
base.POST_NORMALIZE_MEMORY = False
base.POST_NORMALIZE_RECURRENT_STATE = False
base.EXPECTED_PARAMETERS = 13_215_008
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_unitary_branch_rms_frobenius_residual_ce_only_"
    "attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "Raw latent and memory carriers; fixed non-affine RMS only on QKV and "
    "decoder coupling branches; actual per-instance memory Frobenius "
    "measurement normalization; one ungated output correction; fully "
    "attached H16 CE; no count schedule, detach, carrier norm, or damping."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
