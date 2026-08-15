"""Train H16 CE with post-norm only at the recurrent carrier boundary."""
from __future__ import annotations

from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_post_update_full_read_h16_attached_stride16_ce_only_13m as h16


h4 = h16.h4
base = h16.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-"
    "urm-boundary-postnorm-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
INTERMEDIATE_CONTROL_BLOCK_NLL = 3.136612097069275
INTERMEDIATE_CONTROL_STEP50_GRADIENT = 765.1490478515625
INTERMEDIATE_CONTROL_STEP100_GRADIENT = 236.73748779296875
BOUNDARY_RMS_MIN = 0.80
BOUNDARY_RMS_MAX = 1.001
ORIGINAL_PREFLIGHT = base.preflight
ORIGINAL_FINAL_CHECKS = h16.additional_final_checks
ORIGINAL_INTERPRETATION = h16.additional_interpretation_lines


def _rms(values: torch.Tensor) -> torch.Tensor:
    return values.float().square().mean(dim=-1).sqrt()


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    generator = torch.Generator().manual_seed(base.SEED + 15119)
    window = base.sampled_windows(
        training,
        2,
        generator,
        base.TRAIN_WINDOW_LENGTH,
    )
    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        roots = output.prefix_encoded.index_select(1, output.anchor_indices)
        initial_state = model.complex_self_prediction.initialize(roots)
        initial_memory_rms = initial_state.memory.abs().square().mean(
            dim=(-3, -2, -1)
        ).sqrt()
        p_rms = _rms(output.preliminary_states)
        z_rms = _rms(output.full_states)
        memory_rms = output.memory_energy.sqrt()

    checks = {
        "initial_raw_memory_rms_min": float(initial_memory_rms.min()),
        "initial_raw_memory_rms_max": float(initial_memory_rms.max()),
        "unnormalized_p_rms_min": float(p_rms.min()),
        "unnormalized_p_rms_max": float(p_rms.max()),
        "boundary_z_rms_min": float(z_rms.min()),
        "boundary_z_rms_max": float(z_rms.max()),
        "boundary_memory_rms_min": float(memory_rms.min()),
        "boundary_memory_rms_max": float(memory_rms.max()),
    }
    for name in ("boundary_z", "boundary_memory"):
        minimum = checks[f"{name}_rms_min"]
        maximum = checks[f"{name}_rms_max"]
        if minimum < BOUNDARY_RMS_MIN or maximum > BOUNDARY_RMS_MAX:
            raise RuntimeError(
                f"{name} RMS escaped boundary norm range: "
                f"{minimum}..{maximum}"
            )
    if checks["unnormalized_p_rms_max"] >= 0.5:
        raise RuntimeError(
            "decoder-facing P unexpectedly looks post-normalized: "
            f"max RMS {checks['unnormalized_p_rms_max']}"
        )
    return {**values, **checks}


def additional_final_checks(initial_row, final_row, rows):
    checks = ORIGINAL_FINAL_CHECKS(initial_row, final_row, rows)
    rows_by_step = {int(row["step"]): row for row in rows}
    checks["final_block_nll_no_worse_than_intermediate_control"] = (
        final_row["val_block_validation_nll"]
        <= INTERMEDIATE_CONTROL_BLOCK_NLL
    )
    if 50 in rows_by_step:
        checks["step50_gradient_below_intermediate_control"] = (
            rows_by_step[50]["gradient_norm"]
            < INTERMEDIATE_CONTROL_STEP50_GRADIENT
        )
    if 100 in rows_by_step:
        checks["step100_gradient_below_intermediate_control"] = (
            rows_by_step[100]["gradient_norm"]
            < INTERMEDIATE_CONTROL_STEP100_GRADIENT
        )
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    rows_by_step = {int(row["step"]): row for row in rows}
    lines = [
        "## Final-carrier-only boundary post-normalization",
        "",
        "P, innovation, and the raw successor memory/read remained "
        "unnormalized. Fixed non-affine norms were applied only to the "
        "complete successor hidden and memory fields immediately before "
        "they were reused by the next central step.",
        "",
        "| Metric | Boundary-only | Intermediate-norm control |",
        "|---|---:|---:|",
        "| Final block NLL | "
        f"{final_row['val_block_validation_nll']:.6f} | "
        f"{INTERMEDIATE_CONTROL_BLOCK_NLL:.6f} |",
    ]
    if 50 in rows_by_step:
        lines.append(
            "| Step-50 raw global gradient norm | "
            f"{rows_by_step[50]['gradient_norm']:.6f} | "
            f"{INTERMEDIATE_CONTROL_STEP50_GRADIENT:.6f} |"
        )
    if 100 in rows_by_step:
        lines.append(
            "| Step-100 raw global gradient norm | "
            f"{rows_by_step[100]['gradient_norm']:.6f} | "
            f"{INTERMEDIATE_CONTROL_STEP100_GRADIENT:.6f} |"
        )
    return [
        *lines,
        "",
        *ORIGINAL_INTERPRETATION(initial_row, final_row, rows),
    ]


h16.EXPERIMENT_ID = EXPERIMENT_ID
h16.DEFAULT_RECORD = DEFAULT_RECORD
h16.DEFAULT_OUTPUT = DEFAULT_OUTPUT
h4.EXPERIMENT_ID = EXPERIMENT_ID
h4.DEFAULT_RECORD = DEFAULT_RECORD
h4.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.EXPERIMENT_ID = EXPERIMENT_ID
base.DEFAULT_RECORD = DEFAULT_RECORD
base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.NORMALIZE_ACCUMULATED_READS = False
base.POST_NORMALIZE_HIDDEN_READS = False
base.POST_NORMALIZE_MEMORY = False
base.POST_NORMALIZE_RECURRENT_STATE = True
base.POST_NORM_EPS = 1e-6
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_urm_boundary_postnorm_parallel_decode_ce_only_"
    "attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The fully attached H16 recurrence left P, innovation, and the raw "
    "successor unnormalized, then applied fixed non-affine post-RMSNorm "
    "only to the final hidden and complex-memory carrier fields before "
    "their reuse. Sixteen raw P states received equal token CE in one "
    "causal decode; no auxiliary target, detach, EMA, or token feedback "
    "was present."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
