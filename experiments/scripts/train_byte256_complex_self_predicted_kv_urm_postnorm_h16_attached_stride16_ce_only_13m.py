"""Train attached H16 CE with URM-style post-normalized central state."""
from __future__ import annotations

from pathlib import Path

import torch

import train_byte256_complex_self_predicted_kv_post_update_full_read_h16_attached_stride16_ce_only_13m as h16


h4 = h16.h4
base = h16.base
EXPERIMENT_ID = (
    "EXP-20260802-byte256-complex-self-predicted-kv-"
    "urm-postnorm-h16-ce-only-attached-stride16-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
ORIGINAL_PREFLIGHT = base.preflight
ORIGINAL_INTERPRETATION = h16.additional_interpretation_lines


def _rms(values: torch.Tensor) -> torch.Tensor:
    return values.float().square().mean(dim=-1).sqrt()


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    generator = torch.Generator().manual_seed(base.SEED + 14117)
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
        initial_memory_energy = initial_state.memory.abs().square().mean(
            dim=(-3, -2, -1)
        )
        p_rms = _rms(output.preliminary_states)
        z_rms = _rms(output.full_states)
        memory_energy = output.memory_energy

    checks = {
        "initial_memory_rms_min": float(initial_memory_energy.sqrt().min()),
        "initial_memory_rms_max": float(initial_memory_energy.sqrt().max()),
        "postnorm_p_rms_min": float(p_rms.min()),
        "postnorm_p_rms_max": float(p_rms.max()),
        "postnorm_z_rms_min": float(z_rms.min()),
        "postnorm_z_rms_max": float(z_rms.max()),
        "postnorm_memory_rms_min": float(memory_energy.sqrt().min()),
        "postnorm_memory_rms_max": float(memory_energy.sqrt().max()),
    }
    for name in (
        "initial_memory",
        "postnorm_p",
        "postnorm_z",
        "postnorm_memory",
    ):
        minimum = checks[f"{name}_rms_min"]
        maximum = checks[f"{name}_rms_max"]
        if minimum < 0.90 or maximum > 1.001:
            raise RuntimeError(
                f"{name} RMS escaped post-norm range: {minimum}..{maximum}"
            )
    return {**values, **checks}


def additional_interpretation_lines(initial_row, final_row, rows):
    lines = ORIGINAL_INTERPRETATION(initial_row, final_row, rows)
    return [
        "## URM-style central post-normalization",
        "",
        "Count-dependent read scaling was disabled. P, recurrent Z, and the "
        "complete complex memory carrier received fixed non-affine RMS "
        "post-normalization after their respective updates.",
        "",
        *lines,
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
base.POST_NORMALIZE_HIDDEN_READS = True
base.POST_NORMALIZE_MEMORY = True
base.POST_NORM_EPS = 1e-6
base.OBJECTIVE_NAME_OVERRIDE = (
    "sixteen_step_urm_postnorm_parallel_decode_ce_only_attached_stride16"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The fully attached H16 central recurrence used fixed non-affine "
    "post-RMSNorm for P and recurrent Z, complex RMSNorm for the complete "
    "memory carrier, and no count-dependent read scaling. Sixteen P states "
    "received equal token CE in one causal decode; no auxiliary target, "
    "detach, EMA, or token feedback was present."
)
base.preflight = preflight
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
