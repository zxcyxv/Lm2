"""Train four sequential central steps with one parallel four-token CE tape."""
from __future__ import annotations

from pathlib import Path

import torch

from rotlm.training.ha_skew_window import (
    forward_sparse_complex_self_predicted_kv_window,
    sparse_complex_self_predicted_kv_ce_only_fast_loss,
)
import train_byte256_complex_self_predicted_kv_h1_online_mse_ce_13m as base


EXPERIMENT_ID = (
    "EXP-20260801-byte256-complex-self-predicted-kv-"
    "post-update-full-read-h4-ce-only-13m"
)
DEFAULT_RECORD = Path("experiments/records") / EXPERIMENT_ID
DEFAULT_OUTPUT = Path("outputs/experiments") / EXPERIMENT_ID
TRAIN_HORIZONS = 4
TRAIN_BOUNDARIES = len(range(0, base.CONTEXT, base.TRAIN_ANCHOR_STRIDE))
DETACH_CROSS_HORIZON_GRADIENTS = False
JOINT_PREDECESSOR_H2_H4_AGREEMENT = 0.382568359375
ORIGINAL_EVALUATE = base.evaluate
BASE_METRIC_NAMES = base.METRIC_NAMES


def objective(model, window, *, latent_target_model=None):
    if latent_target_model is not None:
        raise ValueError("four-token CE-only training has no target model")
    output = forward_sparse_complex_self_predicted_kv_window(
        model,
        window,
        prefix_length=base.CONTEXT,
        horizons=TRAIN_HORIZONS,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
        detach_cross_horizon_gradients=(
            DETACH_CROSS_HORIZON_GRADIENTS
        ),
    )
    token_ce = output.token_ce.mean()
    no_auxiliary_loss = token_ce.new_zeros(())
    return token_ce, token_ce, no_auxiliary_loss, output


def fast_training_objective(model, window, *, latent_target_model=None):
    if latent_target_model is not None:
        raise ValueError("CE-only training has no target model")
    token_ce = sparse_complex_self_predicted_kv_ce_only_fast_loss(
        model,
        window,
        prefix_length=base.CONTEXT,
        horizons=TRAIN_HORIZONS,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
    )
    no_auxiliary_loss = token_ce.new_zeros(())
    return token_ce, token_ce, no_auxiliary_loss, None


@torch.inference_mode()
def evaluate(model, validation, starts, microbatch, *, latent_target_model=None):
    metrics = ORIGINAL_EVALUATE(
        model,
        validation,
        starts,
        microbatch,
        latent_target_model=latent_target_model,
    )
    metrics["no_auxiliary_loss"] = 0.0
    return metrics


def _require_gradient(name: str, gradient: torch.Tensor | None) -> float:
    if (
        gradient is None
        or not bool(torch.isfinite(gradient).all())
        or not bool(gradient.norm() > 0)
    ):
        raise RuntimeError(f"no finite nonzero H4 CE gradient to {name}")
    return float(gradient.norm())


def _finite_gradient_norm(
    name: str,
    gradient: torch.Tensor | None,
) -> float:
    if gradient is None:
        return 0.0
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError(f"non-finite gradient to {name}")
    return float(gradient.norm())


def preflight(model, training, *, latent_target_model=None):
    if latent_target_model is not None:
        raise RuntimeError("CE-only preflight unexpectedly received EMA")
    generator = torch.Generator().manual_seed(base.SEED + 14117)
    window = base.sampled_windows(
        training,
        2,
        generator,
        base.TRAIN_WINDOW_LENGTH,
    )
    loss, token_ce, auxiliary, output = objective(model, window)
    expected_shape = (2, TRAIN_BOUNDARIES, TRAIN_HORIZONS)
    if output.token_ce.shape != expected_shape:
        raise RuntimeError(
            f"unexpected four-token CE shape {tuple(output.token_ce.shape)}"
        )
    torch.testing.assert_close(loss, output.token_ce.mean())
    torch.testing.assert_close(loss, token_ce)
    if float(auxiliary) != 0.0 or auxiliary.requires_grad:
        raise RuntimeError("CE-only objective registered an auxiliary loss")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite four-token CE")

    target_gradient = torch.autograd.grad(
        loss,
        output.gold_states,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if target_gradient is not None and bool(target_gradient.abs().max() > 0):
        raise RuntimeError("future hidden target influenced CE-only training")

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
    names = (
        "query",
        "key",
        "value",
        "complex readout",
        "hidden phase",
        "memory phase",
        "online encoder final block",
    )
    h4_ce = output.token_ce[:, :, -1].mean()
    h4_gradients = torch.autograd.grad(
        h4_ce,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    h4_gradient_norms = {
        name: _finite_gradient_norm(name, gradient)
        for name, gradient in zip(names, h4_gradients)
    }
    if DETACH_CROSS_HORIZON_GRADIENTS:
        for name in ("key", "value"):
            if h4_gradient_norms[name] != 0.0:
                raise RuntimeError(
                    f"detached H4 CE unexpectedly reached {name} writes"
                )
        for name in (
            "query",
            "complex readout",
            "hidden phase",
            "memory phase",
            "online encoder final block",
        ):
            if not h4_gradient_norms[name] > 0.0:
                raise RuntimeError(
                    f"detached H4 CE did not reach current {name}"
                )
    else:
        for name, gradient in zip(names, h4_gradients):
            _require_gradient(name, gradient)

    full_gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    full_gradient_norms = {
        name: _require_gradient(
            f"full CE {name}",
            gradient,
        )
        for name, gradient in zip(names, full_gradients)
    }
    gradient_norms = {
        **{
            f"h4_ce_{name.replace(' ', '_')}_gradient_norm": value
            for name, value in h4_gradient_norms.items()
        },
        **{
            f"full_ce_{name.replace(' ', '_')}_gradient_norm": value
            for name, value in full_gradient_norms.items()
        },
    }

    detach_checks = {
        "detach_cross_horizon_gradients": (
            DETACH_CROSS_HORIZON_GRADIENTS
        ),
    }
    if DETACH_CROSS_HORIZON_GRADIENTS:
        with torch.no_grad():
            attached_output = (
                forward_sparse_complex_self_predicted_kv_window(
                    model,
                    window,
                    prefix_length=base.CONTEXT,
                    horizons=TRAIN_HORIZONS,
                    anchor_stride=base.TRAIN_ANCHOR_STRIDE,
                    detach_cross_horizon_gradients=False,
                )
            )
        forward_identity_error = max(
            float((left.detach() - right).abs().max())
            for left, right in (
                (output.read_states, attached_output.read_states),
                (output.full_states, attached_output.full_states),
                (output.preliminary_states, attached_output.preliminary_states),
                (output.rotated_states, attached_output.rotated_states),
                (output.innovation_deltas, attached_output.innovation_deltas),
                (output.memory_energy, attached_output.memory_energy),
                (output.logits, attached_output.logits),
            )
        )
        if forward_identity_error >= 5e-5:
            raise RuntimeError(
                "cross-horizon detach changed forward values"
            )

        decoder_state_gradient = torch.autograd.grad(
            h4_ce,
            output.read_states,
            retain_graph=True,
        )[0]
        decoder_past_gradient_max = float(
            decoder_state_gradient[:, :, :-1].abs().max()
        )
        decoder_current_gradient_norm = float(
            decoder_state_gradient[:, :, -1].norm()
        )
        if decoder_past_gradient_max != 0.0:
            raise RuntimeError(
                "H4 CE reached an earlier decoder latent slot"
            )
        if not decoder_current_gradient_norm > 0.0:
            raise RuntimeError("H4 CE did not reach its own decoder slot")

        roots = (
            output.prefix_encoded.detach()
            .index_select(1, output.anchor_indices)
            .requires_grad_(True)
        )
        attached_rollout = transition.rollout(
            roots,
            2,
            detach_state_between_horizons=False,
        )
        attached_h2_loss = attached_rollout.read_states[:, :, 1].square().mean()
        attached_root_gradient = torch.autograd.grad(
            attached_h2_loss,
            roots,
            retain_graph=True,
        )[0]
        attached_root_gradient_norm = float(attached_root_gradient.norm())
        if not attached_root_gradient_norm > 0.0:
            raise RuntimeError("attached H2 read has no root gradient")

        detached_rollout = transition.rollout(
            roots,
            2,
            detach_state_between_horizons=True,
        )
        detached_h2_loss = detached_rollout.read_states[:, :, 1].square().mean()
        detached_root_gradient = torch.autograd.grad(
            detached_h2_loss,
            roots,
            retain_graph=True,
            allow_unused=True,
        )[0]
        detached_root_gradient_norm = (
            0.0
            if detached_root_gradient is None
            else float(detached_root_gradient.norm())
        )
        if detached_root_gradient_norm != 0.0:
            raise RuntimeError("detached H2 read reached the initial root")
        detached_parameter_gradient = torch.autograd.grad(
            detached_h2_loss,
            transition.query.weight,
            retain_graph=True,
        )[0]
        detached_parameter_gradient_norm = _require_gradient(
            "detached H2 local query",
            detached_parameter_gradient,
        )
        detach_checks.update(
            {
                "cross_horizon_detach_forward_max_error": (
                    forward_identity_error
                ),
                "h4_decoder_past_slot_gradient_max": (
                    decoder_past_gradient_max
                ),
                "h4_decoder_current_slot_gradient_norm": (
                    decoder_current_gradient_norm
                ),
                "attached_h2_root_gradient_norm": (
                    attached_root_gradient_norm
                ),
                "detached_h2_root_gradient_norm": (
                    detached_root_gradient_norm
                ),
                "detached_h2_local_query_gradient_norm": (
                    detached_parameter_gradient_norm
                ),
            }
        )

    changed = window.clone()
    changed[:, base.CONTEXT :] = (
        changed[:, base.CONTEXT :] + 1
    ) % base.VOCABULARY
    changed_output = forward_sparse_complex_self_predicted_kv_window(
        model,
        changed,
        prefix_length=base.CONTEXT,
        horizons=TRAIN_HORIZONS,
        anchor_stride=base.TRAIN_ANCHOR_STRIDE,
        detach_cross_horizon_gradients=(
            DETACH_CROSS_HORIZON_GRADIENTS
        ),
    )
    future_input_error = max(
        float((left - right).detach().abs().max())
        for left, right in (
            (output.read_states, changed_output.read_states),
            (output.full_states, changed_output.full_states),
            (output.logits, changed_output.logits),
        )
    )
    if future_input_error != 0.0:
        raise RuntimeError("held-out future byte leaked into four-step tape")
    if torch.equal(output.targets, changed_output.targets):
        raise RuntimeError("future-byte mutation did not change CE labels")

    perturbed_states = output.read_states.detach().clone()
    perturbed_states[:, :, -1].add_(0.125)
    perturbed_hidden = model.exact_inverse_selected_decode_tape_states(
        output.prefix_encoded,
        perturbed_states,
        output.full_positions[: base.CONTEXT],
        output.anchor_indices,
    )
    perturbed_logits = model.token_logits(perturbed_hidden)
    causal_prefix_error = (
        float(
            (
                perturbed_logits[:, :, :-1]
                - output.logits[:, :, :-1]
            )
            .detach()
            .abs()
            .max()
        )
        if TRAIN_HORIZONS > 1
        else 0.0
    )
    causal_h4_delta = float(
        (
            perturbed_logits[:, :, -1]
            - output.logits[:, :, -1]
        ).detach().abs().mean()
    )
    if causal_prefix_error >= 1e-5 or causal_h4_delta <= 0.0:
        raise RuntimeError("parallel decoder tape is not causally triangular")

    if transition.beta is not None:
        raise RuntimeError("CE-only block run unexpectedly has beta")
    if transition.residual_prior:
        raise RuntimeError("CE-only block run unexpectedly has residual prior")
    if transition.recurrent_hidden_mode != base.RECURRENT_HIDDEN_MODE:
        raise RuntimeError("CE-only block run uses the wrong recurrence")
    if (
        transition.pre_normalize_qkv_inputs
        != base.PRE_NORMALIZE_QKV_INPUTS
    ):
        raise RuntimeError("CE-only block run has the wrong QKV pre-norm")
    if (
        transition.normalize_initial_recurrent_root
        != base.NORMALIZE_INITIAL_RECURRENT_ROOT
    ):
        raise RuntimeError(
            "CE-only block run has the wrong initial recurrent-root norm"
        )
    if (
        transition.normalize_accumulated_reads
        != base.NORMALIZE_ACCUMULATED_READS
    ):
        raise RuntimeError("CE-only block run has the wrong read scaling")
    if (
        transition.post_normalize_hidden_reads
        != base.POST_NORMALIZE_HIDDEN_READS
    ):
        raise RuntimeError("CE-only block run has the wrong hidden post-norm")
    if transition.post_normalize_memory != base.POST_NORMALIZE_MEMORY:
        raise RuntimeError("CE-only block run has the wrong memory post-norm")
    if (
        transition.post_normalize_recurrent_state
        != base.POST_NORMALIZE_RECURRENT_STATE
    ):
        raise RuntimeError(
            "CE-only block run has the wrong final-carrier post-norm"
        )
    total_parameters = base.parameter_count(model)
    if total_parameters != base.EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"parameter total {total_parameters} != {base.EXPECTED_PARAMETERS}"
        )
    model.zero_grad(set_to_none=True)
    return {
        "initial_loss": float(loss.detach()),
        "initial_token_ce": float(token_ce.detach()),
        "initial_closure_loss": 0.0,
        "initial_latent_relative_mse": float(
            output.latent_relative_mse.mean().detach()
        ),
        "future_hidden_target_gradient_norm": 0.0,
        "future_input_leak_error": future_input_error,
        "parallel_decoder_causal_prefix_error": causal_prefix_error,
        "parallel_decoder_h4_perturbation_delta": causal_h4_delta,
        **detach_checks,
        **gradient_norms,
    }


def additional_final_checks(initial_row, final_row, rows):
    del rows
    checks = {
        "block_nll_improved": final_row["val_block_validation_nll"]
        < initial_row["val_block_validation_nll"],
        "block_nll_below_3": final_row["val_block_validation_nll"] < 3.0,
        "h2_h4_agreement_improved_over_joint_predecessor": final_row[
            "val_h2_h4_token_agreement"
        ]
        > JOINT_PREDECESSOR_H2_H4_AGREEMENT,
    }
    for horizon in range(1, TRAIN_HORIZONS + 1):
        checks[f"h{horizon}_nll_improved"] = final_row[
            f"val_h{horizon}_target_nll"
        ] < initial_row[f"val_h{horizon}_target_nll"]
        checks[f"h{horizon}_accuracy_above_random"] = final_row[
            f"val_h{horizon}_target_accuracy"
        ] > 1.0 / base.VOCABULARY
    return checks


def additional_interpretation_lines(initial_row, final_row, rows):
    del rows
    lines = [
        "## Four-token CE block",
        "",
        "The central layer executed four sequential continuous transitions. "
        "The exact-inverse decoder consumed the four P states as one causal "
        "tape and produced all four training logits in one batched call.",
        "",
        "| Horizon | Initial NLL | Final NLL | Final target accuracy | Final central/AR agreement |",
        "|---:|---:|---:|---:|---:|",
    ]
    for horizon in range(1, TRAIN_HORIZONS + 1):
        lines.append(
            f"| H{horizon} | "
            f"{initial_row[f'val_h{horizon}_target_nll']:.6f} | "
            f"{final_row[f'val_h{horizon}_target_nll']:.6f} | "
            f"{final_row[f'val_h{horizon}_target_accuracy']:.6f} | "
            f"{final_row[f'val_h{horizon}_token_agreement']:.6f} |"
        )
    lines.extend(
        (
            "",
            f"Four-token mean validation NLL: {initial_row['val_block_validation_nll']:.6f} -> {final_row['val_block_validation_nll']:.6f}.",
            "",
        )
    )
    return lines


base.EXPERIMENT_ID = EXPERIMENT_ID
base.DEFAULT_RECORD = DEFAULT_RECORD
base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.TEMPERED_READOUT = False
base.RESIDUAL_PRIOR = False
base.RECURRENT_HIDDEN_MODE = "post-update-full-read"
base.NORMALIZE_ACCUMULATED_READS = True
base.TRAIN_HORIZONS = TRAIN_HORIZONS
base.TRAIN_WINDOW_LENGTH = base.CONTEXT + TRAIN_HORIZONS
base.TRAIN_ANCHORS = TRAIN_BOUNDARIES * TRAIN_HORIZONS
base.EMA_TARGET_DECAY = None
base.EMA_WARM_START_STEPS = 0
base.EVALUATE_WITH_EMA = False
base.REPORT_ONLINE_WITH_EMA = False
base.REPORT_STEPS = frozenset(
    (1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
)
base.EXPECTED_PARAMETERS = 13_216_352
base.MIN_H2_H4_TOKEN_AGREEMENT = 0.0
base.MIN_MEAN_AR_STATE_COSINE = -1.0
base.MIN_H1_POSTERIOR_TARGET_ACCURACY = None
base.STEP100_REFERENCE = None
base.LATENT_WEIGHT = 0.0
base.DETACH_LATENT_TARGET = False
base.CLOSURE_METRIC_NAME = "no_auxiliary_loss"
base.CLOSURE_DISPLAY_NAME = "aux"
base.CLOSURE_VALIDATION_METRIC_NAME = "no_auxiliary_loss"
base.OBJECTIVE_NAME_OVERRIDE = "four_step_central_parallel_decode_ce_only"
base.TRAIN_OBJECTIVE_OVERRIDE = fast_training_objective
base.CLOSURE_TARGET_NAME_OVERRIDE = "none"
base.REQUIRE_H1_LATENT_MSE_IMPROVEMENT = False
base.REQUIRE_WRITE_EXCEEDS_NO_WRITE = False
base.TRAINING_TARGET_DESCRIPTION_OVERRIDE = (
    "The central recurrence executed four sequential target-free steps. Its "
    "four innovation-free P states were decoded as one causal tape and all "
    "four corpus-future bytes supplied CE labels only. No hidden, KL, EMA, "
    "or token-reencoding target was present."
)
base.objective = objective
base.preflight = preflight
base.evaluate = evaluate
base.METRIC_NAMES = (*BASE_METRIC_NAMES, "no_auxiliary_loss")
base.additional_final_checks = additional_final_checks
base.additional_interpretation_lines = additional_interpretation_lines


if __name__ == "__main__":
    base.main()
