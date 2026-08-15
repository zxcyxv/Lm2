"""Ten-epoch H4-supervised feedback recurrence with H16 monitoring."""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from rotlm.training.ha_skew_window import (
    forward_sparse_complex_self_predicted_kv_window,
)
import train_byte256_complex_self_predicted_kv_unitary_branch_normalized_residual_h16_attached_stride16_ce_only_13m as run


base = run.base
h4 = run.h4

EXPERIMENT_ID = (
    "EXP-20260805-byte256-unitary-feedback-h4-stride4-h16-monitor-"
    "10epoch-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

TRAIN_HORIZONS = 4
TRAIN_ANCHOR_STRIDE = 4
MONITOR_HORIZONS = 16
EVAL_ANCHOR_STRIDE = 16
EPOCH_STEPS = 3_357
TOTAL_STEPS = 10 * EPOCH_STEPS
REPORT_STEPS = frozenset(
    (1, 50, 100, 200, 300, 1_000)
    + tuple(EPOCH_STEPS * epoch for epoch in range(1, 11))
)
ORIGINAL_PREFLIGHT = run.preflight
MASKED_GRADIENT_RELATIVE_TOLERANCE = 2e-4


def preflight(model, training, *, latent_target_model=None):
    values = ORIGINAL_PREFLIGHT(
        model,
        training,
        latent_target_model=latent_target_model,
    )
    if latent_target_model is not None:
        raise RuntimeError("CE-only H4 experiment unexpectedly received EMA")

    generator = torch.Generator().manual_seed(base.SEED + 28_805)
    full_window = base.sampled_windows(
        training,
        1,
        generator,
        base.CONTEXT + MONITOR_HORIZONS,
    )
    truncated_window = full_window[:, : base.CONTEXT + TRAIN_HORIZONS]
    full = forward_sparse_complex_self_predicted_kv_window(
        model,
        full_window,
        prefix_length=base.CONTEXT,
        horizons=MONITOR_HORIZONS,
        anchor_stride=TRAIN_ANCHOR_STRIDE,
    )
    truncated = forward_sparse_complex_self_predicted_kv_window(
        model,
        truncated_window,
        prefix_length=base.CONTEXT,
        horizons=TRAIN_HORIZONS,
        anchor_stride=TRAIN_ANCHOR_STRIDE,
    )

    if not torch.equal(full.targets[..., :TRAIN_HORIZONS], truncated.targets):
        raise RuntimeError("H16 and truncated-H4 targets differ")
    if not torch.equal(full.anchor_indices, truncated.anchor_indices):
        raise RuntimeError("H16 and truncated-H4 anchors differ")
    forward_max_error = max(
        float(
            (left[..., :TRAIN_HORIZONS, :] - right)
            .detach()
            .abs()
            .max()
        )
        for left, right in (
            (full.logits, truncated.logits),
            (full.read_states, truncated.read_states),
            (full.full_states, truncated.full_states),
        )
    )
    forward_tolerance = 5e-5 * max(1.0, model.width / 1344)
    if forward_max_error >= forward_tolerance:
        raise RuntimeError(
            "causal H4 dead-code elimination changed forward values: "
            f"{forward_max_error}"
        )

    masked_full_loss = F.cross_entropy(
        full.logits[..., :TRAIN_HORIZONS, :].reshape(-1, base.VOCABULARY).float(),
        full.targets[..., :TRAIN_HORIZONS].reshape(-1),
    )
    truncated_loss = truncated.token_ce.mean()
    torch.testing.assert_close(masked_full_loss, truncated_loss)

    full_logit_gradient = torch.autograd.grad(
        masked_full_loss,
        full.logits,
        retain_graph=True,
    )[0]
    supervised_logit_gradient_norm = float(
        full_logit_gradient[..., :TRAIN_HORIZONS, :].norm()
    )
    heldout_logit_gradient_max = float(
        full_logit_gradient[..., TRAIN_HORIZONS:, :].abs().max()
    )
    if supervised_logit_gradient_norm <= 0.0:
        raise RuntimeError("masked H1-H4 logits received no gradient")
    if heldout_logit_gradient_max != 0.0:
        raise RuntimeError("masked H5-H16 logits received a gradient")

    transition = model.complex_self_prediction
    parameters = (
        transition.query.weight,
        transition.key.weight,
        transition.value.weight,
        transition.output.weight,
        transition.hidden_phase,
        transition.memory_phase,
        model.encoder.blocks[-1].attn.qkv.weight,
    )
    full_gradients = torch.autograd.grad(
        masked_full_loss,
        parameters,
        retain_graph=True,
    )
    truncated_gradients = torch.autograd.grad(truncated_loss, parameters)
    parameter_gradient_max_error = max(
        float((full_gradient - truncated_gradient).abs().max())
        for full_gradient, truncated_gradient in zip(
            full_gradients,
            truncated_gradients,
        )
    )
    parameter_gradient_max_scale = max(
        float(full_gradient.abs().max()) for full_gradient in full_gradients
    )
    parameter_gradient_relative_error = parameter_gradient_max_error / max(
        parameter_gradient_max_scale,
        1e-12,
    )
    if parameter_gradient_relative_error >= MASKED_GRADIENT_RELATIVE_TOLERANCE:
        raise RuntimeError(
            "masked-H16 and truncated-H4 parameter gradients differ: "
            f"{parameter_gradient_relative_error}"
        )
    model.zero_grad(set_to_none=True)
    return {
        **values,
        "masked_h16_truncated_h4_forward_max_error": forward_max_error,
        "masked_h16_truncated_h4_loss_abs_error": float(
            (masked_full_loss - truncated_loss).detach().abs()
        ),
        "masked_h1_h4_logit_gradient_norm": supervised_logit_gradient_norm,
        "masked_h5_h16_logit_gradient_max": heldout_logit_gradient_max,
        "masked_h16_truncated_h4_parameter_gradient_max_error": (
            parameter_gradient_max_error
        ),
        "masked_h16_truncated_h4_parameter_gradient_relative_error": (
            parameter_gradient_relative_error
        ),
    }


def additional_final_checks(initial_row, final_row, rows):
    epoch_one = next(
        (row for row in rows if int(row["step"]) == EPOCH_STEPS),
        None,
    )
    checks = {
        "epoch_one_row_present": epoch_one is not None,
        "all_reported_losses_and_gradients_finite": all(
            all(
                math.isfinite(float(row[name]))
                for name in (
                    "train_loss",
                    "gradient_norm",
                    "val_block_validation_nll",
                )
                if not (int(row["step"]) == 0 and name != "val_block_validation_nll")
            )
            for row in rows
        ),
        "h16_monitor_block_nll_improved_from_initialization": final_row[
            "val_block_validation_nll"
        ]
        < initial_row["val_block_validation_nll"],
    }
    for horizon in range(1, TRAIN_HORIZONS + 1):
        checks[f"supervised_h{horizon}_nll_improved"] = final_row[
            f"val_h{horizon}_target_nll"
        ] < initial_row[f"val_h{horizon}_target_nll"]
    if epoch_one is not None:
        heldout_improvements = [
            epoch_one[f"val_h{horizon}_target_nll"]
            - final_row[f"val_h{horizon}_target_nll"]
            for horizon in range(TRAIN_HORIZONS + 1, MONITOR_HORIZONS + 1)
        ]
        checks["heldout_h5_h16_mean_nll_improved_after_epoch_one"] = (
            sum(heldout_improvements) / len(heldout_improvements) > 0.0
        )
        checks["heldout_h5_h16_majority_improved_after_epoch_one"] = (
            sum(value > 0.0 for value in heldout_improvements)
            >= math.ceil(0.75 * len(heldout_improvements))
        )
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    del rows
    lines = [
        "## H4-supervised feedback recurrence with H16 monitoring",
        "",
        "Training unfolded the state-dependent feedback recurrence through H4 "
        "and backpropagated equal CE from H1--H4 only. Validation unfolded the "
        "same shared transition through H16; H5--H16 therefore measure "
        "out-of-objective rollout transfer.",
        "",
        "| Horizon | Supervised in training | Initial NLL | Final NLL |",
        "|---:|:---:|---:|---:|",
    ]
    for horizon in range(1, MONITOR_HORIZONS + 1):
        lines.append(
            f"| H{horizon} | {'yes' if horizon <= TRAIN_HORIZONS else 'no'} | "
            f"{initial_row[f'val_h{horizon}_target_nll']:.6f} | "
            f"{final_row[f'val_h{horizon}_target_nll']:.6f} |"
        )
    return lines


for module in (run, run.parent, run.h16, h4, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.DEFAULT_RECORD = DEFAULT_RECORD
    module.DEFAULT_OUTPUT = DEFAULT_OUTPUT

h4.TRAIN_HORIZONS = TRAIN_HORIZONS
h4.TRAIN_BOUNDARIES = len(range(0, base.CONTEXT, TRAIN_ANCHOR_STRIDE))
h4.DETACH_CROSS_HORIZON_GRADIENTS = False

base.HORIZONS = MONITOR_HORIZONS
base.TRAIN_HORIZONS = TRAIN_HORIZONS
base.TRAIN_ANCHOR_STRIDE = TRAIN_ANCHOR_STRIDE
base.EVAL_ANCHOR_STRIDE = EVAL_ANCHOR_STRIDE
base.TRAIN_WINDOW_LENGTH = base.CONTEXT + TRAIN_HORIZONS
base.EVAL_WINDOW_LENGTH = base.CONTEXT + MONITOR_HORIZONS
base.TRAIN_ANCHORS = h4.TRAIN_BOUNDARIES * TRAIN_HORIZONS
base.METRIC_NAMES = (
    *base.metric_names_for_horizons(MONITOR_HORIZONS),
    "no_auxiliary_loss",
)
base.STEPS = TOTAL_STEPS
base.SCHEDULE_STEPS = TOTAL_STEPS
base.EFFECTIVE_BATCH = 64
base.MICROBATCH = 64
base.REPORT_STEPS = REPORT_STEPS
base.OBJECTIVE_NAME_OVERRIDE = (
    "four_step_unitary_feedback_branch_rms_frobenius_residual_ce_only_"
    "attached_stride4_h16_monitor"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The state-dependent H16 feedback architecture used equal H1--H4 token "
    "CE as its only loss. Causally downstream H5--H16 training nodes were "
    "dead-code-eliminated after preflight proved their removal preserved the "
    "selected loss and parameter gradients. Validation unfolded full H16."
)
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
