"""Run the 121M H4 feedback model without the hidden residual addition."""
from __future__ import annotations

import math
from pathlib import Path

import torch

import train_byte256_unitary_feedback_h4_stride4_h16_monitor_10epoch_121m as parent


base = parent.base
EXPERIMENT_ID = (
    "EXP-20260807-byte256-unitary-feedback-h4-width4352-121m-lr2e-4-"
    "no-hidden-residual-500step"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
RECURRENT_HIDDEN_MODE = "unitary-branch-normalized-no-residual"
PEAK_LR = 2e-4
STEPS = 500
SCHEDULE_STEPS = 33_570
EXPECTED_PARAMETERS = 121_336_064


GENERIC_PREFLIGHT = parent.run.run.ORIGINAL_PREFLIGHT


def preflight(model, training, *, latent_target_model=None):
    values = GENERIC_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    if latent_target_model is not None:
        raise RuntimeError("CE-only no-residual run unexpectedly received EMA")
    transition = model.complex_self_prediction
    if transition.recurrent_hidden_mode != RECURRENT_HIDDEN_MODE:
        raise RuntimeError("no-residual branch mode is disabled")

    generator = torch.Generator().manual_seed(base.SEED + 31_707)
    window = base.sampled_windows(
        training, 1, generator, base.TRAIN_WINDOW_LENGTH
    )
    with torch.no_grad():
        _, _, _, output = base.objective(model, window)
        roots = output.prefix_encoded.index_select(1, output.anchor_indices)
        initial = transition.initialize(roots)
        first = transition(initial)
        second = transition(first.state)
        expected_decode = transition._fixed_rms_normalize_hidden(
            first.innovation_delta
        )

    diagnostics = {
        "no_residual_first_hidden_max_error": float(
            (first.state.hidden - first.innovation_delta).abs().max()
        ),
        "no_residual_second_hidden_max_error": float(
            (second.state.hidden - second.innovation_delta).abs().max()
        ),
        "no_residual_first_decode_max_error": float(
            (first.read_hidden - expected_decode).abs().max()
        ),
        "removed_residual_first_gap": float(
            (
                first.state.hidden
                - (first.rotated_hidden + first.innovation_delta)
            ).abs().max()
        ),
        "zero_initial_memory_max_abs": float(initial.memory.abs().max()),
        "parameter_count_no_residual": base.parameter_count(model),
    }
    for name in (
        "no_residual_first_hidden_max_error",
        "no_residual_second_hidden_max_error",
        "no_residual_first_decode_max_error",
        "zero_initial_memory_max_abs",
    ):
        if diagnostics[name] != 0.0:
            raise RuntimeError(f"no-residual formula mismatch: {name}")
    if diagnostics["removed_residual_first_gap"] <= 1e-3:
        raise RuntimeError("hidden residual was not observably removed")
    if diagnostics["parameter_count_no_residual"] != EXPECTED_PARAMETERS:
        raise RuntimeError("no-residual mode changed the parameter count")
    return {**values, **diagnostics}


def additional_final_checks(initial_row, final_row, rows):
    del initial_row
    return {
        "step500_row_present": int(final_row["step"]) == 500,
        "reported_losses_and_gradients_finite": all(
            all(
                math.isfinite(float(row[name]))
                for name in (
                    "train_loss",
                    "gradient_norm",
                    "val_block_validation_nll",
                )
                if int(row["step"]) != 0 or name == "val_block_validation_nll"
            )
            for row in rows
        ),
    }


for module in (
    parent,
    parent.run,
    parent.run.run,
    parent.run.run.parent,
    parent.run.h4,
    base,
):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

parent.PEAK_LR = PEAK_LR
base.PEAK_LR = PEAK_LR
base.RECURRENT_HIDDEN_MODE = RECURRENT_HIDDEN_MODE
base.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
base.STEPS = STEPS
base.SCHEDULE_STEPS = SCHEDULE_STEPS
base.REPORT_STEPS = frozenset(
    set(base.REPORT_STEPS) | {1, 10, 25, 50, 100, 200, 300, 500}
)
base.OBJECTIVE_NAME_OVERRIDE = (
    "four_step_unitary_feedback_branch_rms_frobenius_no_hidden_residual_"
    "ce_only_attached_stride4_h16_monitor"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The registered 121M LR-2e-4 feedback recurrence with only the hidden "
    "successor addition removed: z_next equals the Frobenius-normalized "
    "measurement delta rather than rotated_z plus that delta."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks


if __name__ == "__main__":
    base.main()
