"""Normalize only the private initial recurrent root of the H16 control."""
from __future__ import annotations

import math
from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_no_qkv_prenorm_h16_attached_stride16_ce_only_13m as control


boundary = control.boundary
h16 = control.h16
h4 = control.h4
base = control.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-postnorm-"
    "no-qkv-prenorm-initial-root-rmsnorm-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
ORIGINAL_PREFLIGHT = control.preflight
ORIGINAL_FINAL_CHECKS = control.additional_final_checks
# Skip the no-QKV wrapper's historical "raw initial carrier" paragraph: the
# matched control already has no QKV pre-norm, while this wrapper changes that
# private initial carrier by construction. Retain the lower boundary/H16
# architecture context instead.
ORIGINAL_INTERPRETATION = control.ORIGINAL_INTERPRETATION


def _rms(values: torch.Tensor) -> torch.Tensor:
    return values.float().square().mean(dim=-1).sqrt()


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    transition = model.complex_self_prediction
    if not transition.normalize_initial_recurrent_root:
        raise RuntimeError("initial recurrent-root RMS normalization is disabled")
    if transition.pre_normalize_qkv_inputs:
        raise RuntimeError("QKV pre-normalization was unexpectedly enabled")
    if not isinstance(transition.input_norm, torch.nn.Identity):
        raise RuntimeError("disabled QKV pre-normalization is not Identity")

    generator = torch.Generator().manual_seed(base.SEED + 17143)
    window = base.sampled_windows(
        training,
        2,
        generator,
        base.TRAIN_WINDOW_LENGTH,
    )
    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        raw_roots = output.prefix_encoded.index_select(
            1,
            output.anchor_indices,
        )
        state = transition.initialize(raw_roots)
        expected = raw_roots.float() * torch.rsqrt(
            raw_roots.float().square().mean(dim=-1, keepdim=True)
            + transition.post_norm_eps
        )
        expected = expected.to(raw_roots.dtype)
        expected_memory = transition.write(expected)

    root_error = float((state.hidden - expected).abs().max())
    memory_error = float((state.memory - expected_memory).abs().max())
    if root_error != 0.0:
        raise RuntimeError(f"initialized Z0 differs from fixed RMS: {root_error}")
    if memory_error != 0.0:
        raise RuntimeError(f"initialized S0 differs from write(Z0): {memory_error}")

    flat_raw = raw_roots.float().reshape(-1, raw_roots.shape[-1])
    flat_root = state.hidden.float().reshape(-1, state.hidden.shape[-1])
    cosine = torch.nn.functional.cosine_similarity(flat_raw, flat_root, dim=-1)
    root_rms = _rms(state.hidden)
    raw_rms = _rms(raw_roots)
    scale = root_rms / raw_rms.clamp_min(torch.finfo(raw_rms.dtype).tiny)
    diagnostics = {
        "initial_root_formula_max_abs_error": root_error,
        "initial_memory_formula_max_abs_error": memory_error,
        "initial_raw_root_rms_min": float(raw_rms.min()),
        "initial_raw_root_rms_max": float(raw_rms.max()),
        "initial_normalized_root_rms_min": float(root_rms.min()),
        "initial_normalized_root_rms_max": float(root_rms.max()),
        "initial_root_positive_scale_min": float(scale.min()),
        "initial_root_positive_scale_max": float(scale.max()),
        "initial_root_cosine_min": float(cosine.min()),
        "initial_root_cosine_max": float(cosine.max()),
        "initial_memory_abs_max": float(state.memory.abs().max()),
        "parameter_count_after_root_norm": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    if not all(
        math.isfinite(float(value)) for value in diagnostics.values()
    ):
        raise RuntimeError("non-finite initial-root normalization diagnostic")
    if diagnostics["initial_normalized_root_rms_min"] < 0.999:
        raise RuntimeError("normalized initial root RMS is materially below one")
    if diagnostics["initial_normalized_root_rms_max"] > 1.001:
        raise RuntimeError("normalized initial root RMS is materially above one")
    if diagnostics["initial_root_positive_scale_min"] <= 0.0:
        raise RuntimeError("initial-root normalization changed a row sign")
    if diagnostics["initial_root_cosine_min"] < 0.999999:
        raise RuntimeError("initial-root normalization changed row direction")
    if diagnostics["parameter_count_after_root_norm"] != base.EXPECTED_PARAMETERS:
        raise RuntimeError("initial-root normalization changed parameter count")
    return {**values, **diagnostics}


def additional_final_checks(initial_row, final_row, rows):
    checks = ORIGINAL_FINAL_CHECKS(initial_row, final_row, rows)
    required_steps = (50, 100, 300, 500, 900, 1000)
    rows_by_step = {int(row["step"]): row for row in rows}
    checks["required_diagnostic_steps_present"] = all(
        step in rows_by_step for step in required_steps
    )
    checks["required_gradient_norms_finite"] = all(
        step not in rows_by_step
        or math.isfinite(float(rows_by_step[step]["gradient_norm"]))
        for step in required_steps
    )
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    return [
        "## Initial recurrent-root normalization",
        "",
        "Only the private encoder-root copy used to initialize recurrent Z0 "
        "and S0 was fixed-RMS-normalized. The raw encoder representation and "
        "exact inverse decoder path were preserved; every post-initialization "
        "transition remained identical to the matched control.",
        "",
        *ORIGINAL_INTERPRETATION(initial_row, final_row, rows),
    ]


for module in (control, boundary, h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.NORMALIZE_INITIAL_RECURRENT_ROOT = True
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_initial_root_rmsnorm_urm_boundary_postnorm_no_qkv_"
    "prenorm_parallel_decode_ce_only_attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The matched no-QKV-pre-norm final-carrier-boundary H16 recurrence was "
    "retained exactly, except a fixed non-affine RMS normalization is applied "
    "to the private encoder-root copy before it initializes both Z0 and the "
    "first KV write S0. Raw encoder and exact inverse paths remain unchanged. "
    "The fully attached loss is sixteen equal token CEs with no auxiliary "
    "target, count scaling, detach, EMA, beta, or token feedback."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
