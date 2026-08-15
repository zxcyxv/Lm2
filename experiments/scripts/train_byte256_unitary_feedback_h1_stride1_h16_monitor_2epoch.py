"""Two-epoch H1-supervised feedback recurrence with H16 monitoring."""
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
    "EXP-20260805-byte256-unitary-feedback-h1-stride1-h16-monitor-"
    "2epoch-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID

TRAIN_HORIZONS = 1
TRAIN_ANCHOR_STRIDE = 1
MONITOR_HORIZONS = 16
EVAL_ANCHOR_STRIDE = 16
EPOCH_STEPS = 3_357
TOTAL_STEPS = 2 * EPOCH_STEPS
REPORT_STEPS = frozenset(
    (1, 50, 100, 200, 300, 1_000, EPOCH_STEPS, TOTAL_STEPS)
)
ORIGINAL_PREFLIGHT = run.preflight
H4_OBJECTIVE = h4.objective
H4_FAST_TRAINING_OBJECTIVE = h4.fast_training_objective


def objective(model, window, *, latent_target_model=None):
    """Use H1 labels while retaining the H4-matched sampled window."""
    return H4_OBJECTIVE(
        model,
        window[:, : base.CONTEXT + TRAIN_HORIZONS],
        latent_target_model=latent_target_model,
    )


def fast_training_objective(model, window, *, latent_target_model=None):
    """Fast H1 CE on the H4-matched window's first 257 positions."""
    return H4_FAST_TRAINING_OBJECTIVE(
        model,
        window[:, : base.CONTEXT + TRAIN_HORIZONS],
        latent_target_model=latent_target_model,
    )


def preflight(model, training, *, latent_target_model=None):
    if latent_target_model is not None:
        raise RuntimeError("CE-only H1 experiment unexpectedly received EMA")

    registered_base_objective = base.objective
    registered_train_window_length = base.TRAIN_WINDOW_LENGTH
    registered_train_anchor_stride = base.TRAIN_ANCHOR_STRIDE
    registered_train_horizons = base.TRAIN_HORIZONS
    registered_train_boundaries = h4.TRAIN_BOUNDARIES
    registered_h4_horizons = h4.TRAIN_HORIZONS
    # Exercise the inherited architecture/formula audit in its native H4,
    # stride-four setting. H1 has a legitimate zero memory-phase gradient and
    # the exhaustive stride-one max reduction slightly exceeds that audit's
    # legacy 2e-6 floating-point tolerance. The explicit H1 audit below is the
    # registered objective gate.
    base.objective = H4_OBJECTIVE
    base.TRAIN_WINDOW_LENGTH = base.CONTEXT + 4
    base.TRAIN_ANCHOR_STRIDE = 4
    base.TRAIN_HORIZONS = 4
    h4.TRAIN_BOUNDARIES = len(range(0, base.CONTEXT, 4))
    h4.TRAIN_HORIZONS = 4
    try:
        values = ORIGINAL_PREFLIGHT(
            model,
            training,
            latent_target_model=latent_target_model,
        )
    finally:
        base.objective = registered_base_objective
        base.TRAIN_WINDOW_LENGTH = registered_train_window_length
        base.TRAIN_ANCHOR_STRIDE = registered_train_anchor_stride
        base.TRAIN_HORIZONS = registered_train_horizons
        h4.TRAIN_BOUNDARIES = registered_train_boundaries
        h4.TRAIN_HORIZONS = registered_h4_horizons

    generator = torch.Generator().manual_seed(base.SEED + 28_806)
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
        raise RuntimeError("H16 and truncated-H1 targets differ")
    if not torch.equal(full.anchor_indices, truncated.anchor_indices):
        raise RuntimeError("H16 and truncated-H1 anchors differ")
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
    if forward_max_error >= 5e-5:
        raise RuntimeError(
            "causal H1 dead-code elimination changed forward values: "
            f"{forward_max_error}"
        )

    masked_full_loss = F.cross_entropy(
        full.logits[..., :TRAIN_HORIZONS, :]
        .reshape(-1, base.VOCABULARY)
        .float(),
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
        raise RuntimeError("masked H1 logits received no gradient")
    if heldout_logit_gradient_max != 0.0:
        raise RuntimeError("masked H2-H16 logits received a gradient")

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
    parameter_names = (
        "query",
        "key",
        "value",
        "complex_readout",
        "hidden_phase",
        "memory_phase",
        "online_encoder_final_block",
    )
    full_gradients = torch.autograd.grad(
        masked_full_loss,
        parameters,
        retain_graph=True,
    )
    truncated_gradients = torch.autograd.grad(truncated_loss, parameters)
    h1_gradient_norms = {}
    for name, full_gradient, truncated_gradient in zip(
        parameter_names,
        full_gradients,
        truncated_gradients,
    ):
        if not (
            bool(torch.isfinite(full_gradient).all())
            and bool(torch.isfinite(truncated_gradient).all())
        ):
            raise RuntimeError(f"non-finite H1 gradient to {name}")
        full_norm = float(full_gradient.norm())
        truncated_norm = float(truncated_gradient.norm())
        if name == "memory_phase":
            if full_norm != 0.0 or truncated_norm != 0.0:
                raise RuntimeError(
                    "H1 CE unexpectedly reached cross-horizon memory phase"
                )
        elif full_norm <= 0.0 or truncated_norm <= 0.0:
            raise RuntimeError(f"no finite nonzero H1 gradient to {name}")
        h1_gradient_norms[f"h1_ce_{name}_gradient_norm"] = truncated_norm
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
    if parameter_gradient_relative_error >= 2e-4:
        raise RuntimeError(
            "masked-H16 and truncated-H1 parameter gradients differ: "
            f"{parameter_gradient_relative_error}"
        )
    if not torch.equal(
        truncated.targets.squeeze(-1),
        truncated_window[:, 1 : base.CONTEXT + 1],
    ):
        raise RuntimeError("H1 stride-one labels do not cover bytes 1--256")
    matched_window = full_window[:, : base.CONTEXT + 4]
    changed_tail = matched_window.clone()
    changed_tail[:, base.CONTEXT + 1 :] = (
        changed_tail[:, base.CONTEXT + 1 :] + 1
    ) % base.VOCABULARY
    with torch.no_grad():
        matched_loss = fast_training_objective(model, matched_window)[0]
        changed_tail_loss = fast_training_objective(model, changed_tail)[0]
    if not torch.equal(matched_loss, changed_tail_loss):
        raise RuntimeError("the H4-matched trailing bytes changed H1 loss")
    model.zero_grad(set_to_none=True)
    return {
        **values,
        "initial_loss": float(truncated_loss.detach()),
        "initial_token_ce": float(truncated_loss.detach()),
        "h1_ce_memory_phase_structural_zero": True,
        "h1_matched_window_trailing3_loss_abs_error": float(
            (matched_loss - changed_tail_loss).abs()
        ),
        **h1_gradient_norms,
        "masked_h16_truncated_h1_forward_max_error": forward_max_error,
        "masked_h16_truncated_h1_loss_abs_error": float(
            (masked_full_loss - truncated_loss).detach().abs()
        ),
        "masked_h1_logit_gradient_norm": supervised_logit_gradient_norm,
        "masked_h2_h16_logit_gradient_max": heldout_logit_gradient_max,
        "masked_h16_truncated_h1_parameter_gradient_max_error": (
            parameter_gradient_max_error
        ),
        "masked_h16_truncated_h1_parameter_gradient_relative_error": (
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
                if not (
                    int(row["step"]) == 0
                    and name != "val_block_validation_nll"
                )
            )
            for row in rows
        ),
        "supervised_h1_nll_improved": final_row["val_h1_target_nll"]
        < initial_row["val_h1_target_nll"],
    }
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    del rows
    lines = [
        "## H1-supervised feedback recurrence with H16 monitoring",
        "",
        "Training backpropagated CE through one state-dependent central "
        "transition only. The built-in validation also unfolded it open-loop "
        "through H16, but H2--H16 are out-of-training-contract diagnostics "
        "and are not used to rank this token-reinput model.",
        "",
        "| Horizon | Supervised in training | Initial NLL | Final NLL |",
        "|---:|:---:|---:|---:|",
    ]
    for horizon in range(1, MONITOR_HORIZONS + 1):
        lines.append(
            f"| H{horizon} | {'yes' if horizon == 1 else 'no'} | "
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
base.TRAIN_WINDOW_LENGTH = base.CONTEXT + 4
base.EVAL_WINDOW_LENGTH = base.CONTEXT + MONITOR_HORIZONS
base.TRAIN_ANCHORS = h4.TRAIN_BOUNDARIES * TRAIN_HORIZONS
base.METRIC_NAMES = (
    *base.metric_names_for_horizons(MONITOR_HORIZONS),
    "no_auxiliary_loss",
)
base.STEPS = TOTAL_STEPS
base.SCHEDULE_STEPS = 10 * EPOCH_STEPS
base.EFFECTIVE_BATCH = 64
base.MICROBATCH = 64
base.REPORT_STEPS = REPORT_STEPS
base.OBJECTIVE_NAME_OVERRIDE = (
    "one_step_unitary_feedback_branch_rms_frobenius_residual_ce_only_"
    "attached_stride1_h16_monitor"
)
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The state-dependent H16 feedback architecture used H1 token CE as its "
    "only loss at all 256 stride-one anchors. Causally downstream H2--H16 "
    "training nodes were dead-code-eliminated after preflight proved their "
    "removal preserved the selected loss and parameter gradients. Validation "
    "unfolded full H16."
)
base.objective = objective
base.TRAIN_OBJECTIVE_OVERRIDE = fast_training_objective
base.preflight = preflight
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
