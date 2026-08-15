"""Ablate only the learned Q/K/V input RMSNorm from boundary-postnorm H16."""
from __future__ import annotations

from pathlib import Path

import math
import torch

import train_byte256_complex_self_predicted_kv_urm_boundary_postnorm_h16_attached_stride16_ce_only_13m as boundary


h16 = boundary.h16
h4 = boundary.h4
base = boundary.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-urm-boundary-"
    "postnorm-no-qkv-prenorm-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
ORIGINAL_PREFLIGHT = boundary.preflight
ORIGINAL_FINAL_CHECKS = boundary.additional_final_checks
ORIGINAL_INTERPRETATION = boundary.additional_interpretation_lines


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    for key in (
        "boundary_z_rms_min",
        "boundary_z_rms_max",
        "boundary_memory_rms_min",
        "boundary_memory_rms_max",
    ):
        if not math.isfinite(float(values[key])):
            raise RuntimeError(f"non-finite boundary diagnostic {key}")
    if float(values["boundary_z_rms_max"]) <= 0.0:
        raise RuntimeError("every initial successor hidden is exactly zero")
    transition = model.complex_self_prediction
    if transition.pre_normalize_qkv_inputs:
        raise RuntimeError("QKV pre-normalization is still enabled")
    if not isinstance(transition.input_norm, torch.nn.Identity):
        raise RuntimeError("disabled QKV pre-norm is not an exact identity")
    if any(
        name.startswith("input_norm")
        for name, _ in transition.named_parameters()
    ):
        raise RuntimeError("disabled QKV pre-norm retained trainable parameters")

    parameter = transition.query.weight
    generator = torch.Generator(device=parameter.device).manual_seed(
        base.SEED + 16127
    )
    probe = torch.randn(
        2,
        3,
        transition.width,
        generator=generator,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    with torch.no_grad():
        scale_error = float(
            (transition.queries(2.0 * probe) - 2.0 * transition.queries(probe))
            .abs()
            .max()
        )
    if scale_error >= 1e-5:
        raise RuntimeError(
            f"Q projection still behaves nonlinearly: {scale_error}"
        )
    return {
        **values,
        "qkv_pre_normalization_disabled": True,
        "q_projection_scale_law_error": scale_error,
    }


def additional_final_checks(initial_row, final_row, rows):
    checks = ORIGINAL_FINAL_CHECKS(initial_row, final_row, rows)
    rows_by_step = {int(row["step"]): row for row in rows}
    if 300 in rows_by_step:
        step300 = rows_by_step[300]
        checks["step300_block_nll_below_divergent_control"] = (
            step300["val_block_validation_nll"] < 223.148930
        )
        checks["step300_gradient_below_divergent_control"] = (
            step300["gradient_norm"] < 604_649_216.0
        )
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    return [
        "## Q/K/V pre-normalization removal",
        "",
        "This run changed only the shared learned RMSNorm immediately before "
        "the central Q/K/V projections. All boundary-postnorm computation, "
        "including the raw initial carrier and raw decoder-facing P, remained "
        "unchanged.",
        "",
        *ORIGINAL_INTERPRETATION(initial_row, final_row, rows),
    ]


for module in (boundary, h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

base.PRE_NORMALIZE_QKV_INPUTS = False
base.EXPECTED_PARAMETERS = 13_215_008
boundary.BOUNDARY_RMS_MIN = 0.0
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_urm_boundary_postnorm_no_qkv_prenorm_parallel_decode_"
    "ce_only_attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The registered final-carrier-only H16 transition was retained exactly, "
    "except Q/K/V now receive their raw hidden inputs without the previous "
    "shared learned RMSNorm. The initial carrier remains raw, P and all "
    "internal innovation/read computations remain unnormalized, and fixed "
    "non-affine normalization is applied only to the complete successor Z/S "
    "carrier. The fully attached objective is sixteen equal token CEs with "
    "no auxiliary target, count scaling, detach, EMA, or token feedback."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
